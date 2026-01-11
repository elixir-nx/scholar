defmodule Scholar.Cluster.OPTICS do
  @moduledoc """
  OPTICS (Ordering Points To Identify the Clustering Structure) is an algorithm
  for finding density-based clusters in spatial data.

  It is closely related to DBSCAN, finds core sample of high density and expands
  clusters from them.  Unlike DBSCAN, keeps cluster hierarchy for a variable
  neighborhood radius.  Clusters are then extracted using a DBSCAN-like
  method.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:labels, :min_samples, :max_eps, :eps, :algorithm]}
  defstruct [:labels, :min_samples, :max_eps, :eps, :algorithm]

  opts = [
    min_samples: [
      default: 5,
      type: :pos_integer,
      doc: """
      The number of samples in a neighborhood for a point to be considered as a core point.
      """
    ],
    max_eps: [
      type: {:custom, Scholar.Options, :beta, []},
      doc: """
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        Default value of Nx.Constants.infinity() will identify clusters across all scales.
      """
    ],
    eps: [
      type: {:custom, Scholar.Options, :beta, []},
      doc: """
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        By default it assumes the same value as max_eps.
      """
    ],
    algorithm: [
      default: :brute,
      type: :atom,
      doc: """
        Algorithm used to compute the k-nearest neighbors. Possible values:

          * `:brute` - Brute-force search. See `Scholar.Neighbors.BruteKNN` for more details.

          * `:kd_tree` - k-d tree. See `Scholar.Neighbors.KDTree` for more details.

          * `:random_projection_forest` - Random projection forest. See `Scholar.Neighbors.RandomProjectionForest` for more details.

          * Module implementing `fit(data, opts)` and `predict(model, query)`. predict/2 must return a tuple containing indices
          of k-nearest neighbors of query points as well as distances between query points and their k-nearest neighbors.
          Also has to take num_neighbors as argument.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Perform OPTICS clustering for `x` which is tensor of `{n_samples, n_features}` shape.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a labels tensor of shape `{n_samples}`.
  Cluster labels for each point in the dataset given to `fit`.
  Noisy samples are labeled as -1.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]])
      iex> Scholar.Cluster.OPTICS.fit(x, min_samples: 2).labels
      #Nx.Tensor<
        s32[6]
        [-1, -1, -1, -1, -1, -1]
      >
      iex> Scholar.Cluster.OPTICS.fit(x, eps: 4.5, min_samples: 2).labels
      #Nx.Tensor<
        s32[6]
        [0, 0, 0, 1, 1, 1]
      >
      iex> Scholar.Cluster.OPTICS.fit(x, eps: 2, min_samples: 2).labels
      #Nx.Tensor<
        s32[6]
        [-1, 0, 0, 1, 1, -1]
      >
      iex> Scholar.Cluster.OPTICS.fit(x, eps: 2, min_samples: 2, algorithm: :kd_tree, metric: {:minkowski, 1}).labels
      #Nx.Tensor<
        s32[6]
        [-1, 0, 0, 1, 1, -1]
      >
      iex> Scholar.Cluster.OPTICS.fit(x, eps: 1, min_samples: 2).labels
      #Nx.Tensor<
        s32[6]
        [-1, -1, -1, 0, 0, -1]
      >
      iex> Scholar.Cluster.OPTICS.fit(x, eps: 4.5, min_samples: 3).labels
      #Nx.Tensor<
        s32[6]
        [0, 0, 0, 1, 1, -1]
      >
  """

  defn fit(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected x to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}
            """
    end

    x = Scholar.Shared.to_float(x)
    module = validate_options(x, opts)

    %__MODULE__{
      module
      | labels: fit_p(x, module)
    }
  end

  deftransformp validate_options(x, opts \\ []) do
    {opts, algorithm_opts} = Keyword.split(opts, [:min_samples, :max_eps, :eps, :algorithm])
    opts = NimbleOptions.validate!(opts, @opts_schema)
    min_samples = opts[:min_samples]

    if min_samples < 2 do
      raise ArgumentError,
            """
            min_samples must be an int in the range [2, inf), got min_samples = #{inspect(min_samples)}
            """
    end

    algorithm_opts = Keyword.put(algorithm_opts, :num_neighbors, 1)

    algorithm_module =
      case opts[:algorithm] do
        :brute ->
          Scholar.Neighbors.BruteKNN

        :kd_tree ->
          Scholar.Neighbors.KDTree

        :random_projection_forest ->
          Scholar.Neighbors.RandomProjectionForest

        module when is_atom(module) ->
          module
      end

    model = algorithm_module.fit(x, algorithm_opts)

    max_eps =
      case opts[:max_eps] do
        nil -> Nx.Constants.infinity(Nx.type(x))
        any -> any
      end

    eps =
      case opts[:eps] do
        nil -> max_eps
        any -> any
      end

    if eps > max_eps do
      raise ArgumentError,
            """
            eps can't be greater than max_eps, got eps = #{inspect(eps)} and max_eps = #{inspect(max_eps)}
            """
    end

    %__MODULE__{
      labels: Nx.broadcast(-1, {Nx.axis_size(x, 0)}),
      min_samples: min_samples,
      max_eps: max_eps,
      eps: eps,
      algorithm: model
    }
  end

  defnp fit_p(x, module) do
    {core_distances, reachability, _predecessor, ordering} = compute_optics_graph(x, module)

    cluster_optics_dbscan(reachability, core_distances, ordering, module)
  end

  defnp compute_optics_graph(x, %__MODULE__{max_eps: max_eps, min_samples: min_samples} = module) do
    n_samples = Nx.axis_size(x, 0)
    reachability = Nx.broadcast(Nx.Constants.max_finite(Nx.type(x)), {n_samples})
    predecessor = Nx.broadcast(-1, {n_samples})
    {_neighbors, distances} = run_knn(x, x, module)
    core_distances = Nx.slice_along_axis(distances, min_samples - 1, 1, axis: 1)

    core_distances =
      Nx.select(core_distances > max_eps, Nx.Constants.infinity(), core_distances)

    ordering = Nx.broadcast(0, {n_samples})
    processed = Nx.broadcast(0, {n_samples})

    {_order_idx, core_distances, reachability, predecessor, _processed, ordering, _x, _module} =
      while {order_idx = 0, core_distances, reachability, predecessor, processed, ordering, x,
             module},
            order_idx < n_samples do
        unprocessed_mask = processed == 0
        point = Nx.argmin(Nx.select(unprocessed_mask, reachability, Nx.Constants.infinity()))
        processed = Nx.put_slice(processed, [point], Nx.new_axis(1, 0))
        ordering = Nx.put_slice(ordering, [order_idx], Nx.new_axis(point, 0))

        {reachability, predecessor} =
          set_reach_dist(core_distances, reachability, predecessor, point, processed, x, module)

        {order_idx + 1, core_distances, reachability, predecessor, processed, ordering, x, module}
      end

    reachability =
      Nx.select(
        reachability == Nx.Constants.max_finite(Nx.type(x)),
        Nx.Constants.infinity(),
        reachability
      )

    {core_distances, reachability, predecessor, ordering}
  end

  defnp set_reach_dist(
          core_distances,
          reachability,
          predecessor,
          point_index,
          processed,
          x,
          %__MODULE__{max_eps: max_eps} = module
        ) do
    n_features = Nx.axis_size(x, 1)
    n_samples = Nx.axis_size(x, 0)
    t = Nx.take(x, point_index, axis: 0)
    p = Nx.broadcast(t, {1, n_features})
    {neighbors, distances} = run_knn(x, p, %__MODULE__{module | min_samples: n_samples})
    neighbors = Nx.flatten(neighbors)
    distances = Nx.flatten(distances)
    indices_ngbrs = Nx.argsort(neighbors)
    neighbors = Nx.take(neighbors, indices_ngbrs)
    distances = Nx.take(distances, indices_ngbrs)
    are_neighbors_processed = Nx.take(processed, neighbors)

    filtered_neighbors =
      Nx.select(
        are_neighbors_processed or distances > max_eps,
        -1 * neighbors,
        neighbors
      )

    dists = Nx.flatten(Scholar.Metrics.Distance.pairwise_minkowski(p, x))
    core_distance = Nx.take(core_distances, point_index)
    rdists = Nx.max(dists, core_distance)
    improved = rdists < reachability
    improved = Nx.select(improved, filtered_neighbors, -1)

    improved =
      Nx.select(
        improved == -1 and filtered_neighbors > 0,
        Nx.multiply(filtered_neighbors, -1),
        filtered_neighbors
      )

    rdists = Nx.select(improved >= 0, rdists, 0)
    reversed_improved = Nx.max(improved * -1, 0)

    reachability =
      Nx.select(improved <= 0, Nx.take(reachability, reversed_improved), rdists)

    predecessor =
      Nx.select(improved <= 0, Nx.take(predecessor, reversed_improved), point_index)

    {reachability, predecessor}
  end

  deftransformp run_knn(x, p, %__MODULE__{algorithm: algorithm_module, min_samples: k} = _module) do
    nbrs = algorithm_module.__struct__.fit(x, num_neighbors: k)
    algorithm_module.__struct__.predict(nbrs, p)
  end

  defnp cluster_optics_dbscan(
          reachability,
          core_distances,
          ordering,
          %__MODULE__{eps: eps} = _module
        ) do
    far_reach = Nx.flatten(reachability > eps)
    near_core = Nx.flatten(core_distances <= eps)
    far_and_not_near = Nx.multiply(far_reach, 1 - near_core)
    far_reach = Nx.take(far_reach, ordering)
    near_core = Nx.take(near_core, ordering)
    far_and_near = far_reach * near_core
    labels = Nx.as_type(Nx.cumulative_sum(far_and_near), :s8) - 1
    labels = Nx.take(labels, Nx.argsort(ordering))
    Nx.select(far_and_not_near, -1, labels)
  end
end
