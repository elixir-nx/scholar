defmodule Scholar.Cluster.OPTICS do
  @moduledoc """
  OPTICS (Ordering Points To Identify the Clustering Structure) is an algorithm
  for finding density-based clusters in spatial data. It is closely related
  to DBSCAN, finds core sample of high density and expands clusters from them. 
  Unlike DBSCAN, keeps cluster hierarchy for a variable neighborhood radius. 
  Clusters are then extracted using a DBSCAN-like method.
  """
  import Nx.Defn
  require Nx

  opts = [
    min_samples: [
      default: 5,
      type: :pos_integer,
      doc: """
      The number of samples in a neighborhood for a point to be considered as a core point.
      """
    ],
    max_eps: [
      default: Nx.Constants.infinity(),
      type: {:custom, Scholar.Options, :beta, []},
      doc: """
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
        Default value of Nx.Constants.infinity() will identify clusters across all scales.
      """
    ],
    eps: [
      default: Nx.Constants.infinity(),
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
  Perform OPTICS clustering for `x` which is tensor of `{n_samples, n_features} shape.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a labels tensor of shape `{n_samples}`.
  Cluster labels for each point in the dataset given to fit().
  Noisy samples are labeled as -1.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]])
      iex> Scholar.Cluster.OPTICS.fit(x, min_samples: 2)
      #Nx.Tensor<
        s64[6]
        [-1, -1, -1, -1, -1, -1]
      >
      iex> Scholar.Cluster.OPTICS.fit(x, eps: 4.5, min_samples: 2)
      #Nx.Tensor<
        s64[6]
        [0, 0, 0, 1, 1, 1]
      >
      iex> Scholar.Cluster.OPTICS.fit(x, eps: 2, min_samples: 2)
      #Nx.Tensor<
        s64[6]
        [-1, 0, 0, 1, 1, -1]
      >
      iex> Scholar.Cluster.OPTICS.fit(x, eps: 1, min_samples: 2)
      #Nx.Tensor<
        s64[6]
        [-1, -1, -1, 0, 0, -1]
      >
      iex> Scholar.Cluster.OPTICS.fit(x, eps: 4.5, min_samples: 3)
      #Nx.Tensor<
        s64[6]
        [0, 0, 0, 1, 1, -1]
      >
      iex> Scholar.Cluster.OPTICS.fit(x, max_eps: 2, min_samples: 1, algorithm: :kd_tree, metric: {:minkowski, 1})
      #Nx.Tensor<
        s64[6]
        [0, 1, 1, 2, 2, 3]
      >
  """

  deftransform fit(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected x to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}
            """
    end
    {opts, algorithm_opts} = Keyword.split(opts, [:min_samples, :max_eps, :eps, :algorithm])
    opts = NimbleOptions.validate!(opts, @opts_schema)
    algorithm_opts = Keyword.put(algorithm_opts, :num_neighbors, opts[:min_samples])

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
    {_neighbors, distances} = algorithm_module.predict(model, x)
    fit_p(x, distances,  opts)
  end

  defnp fit_p(x, core_distances, opts \\ []) do
    {core_distances, reachability, _predecessor, ordering} = compute_optics_graph(x, core_distances, opts)

    eps =
      if opts[:eps] == Nx.Constants.infinity() do
        opts[:max_eps]
      else
        opts[:eps]
      end

    cluster_optics_dbscan(reachability, core_distances, ordering, eps: eps)
  end

  defnp compute_optics_graph(x, distances, opts \\ []) do
    max_eps = opts[:max_eps]
    n_samples = Nx.axis_size(x, 0)
    reachability = Nx.broadcast(Nx.Constants.max_finite({:f, 32}), {n_samples})
    predecessor = Nx.broadcast(-1, {n_samples})
    core_distances = Nx.slice_along_axis(distances, opts[:min_samples] - 1, 1, axis: 1)
    core_distances =
      Nx.select(core_distances > max_eps, Nx.Constants.infinity(), core_distances)

    ordering = Nx.broadcast(0, {n_samples})
    processed = Nx.broadcast(0, {n_samples})

    {_order_idx, core_distances, reachability, predecessor, _processed, ordering, _x, _max_eps} =
      while {order_idx = 0, core_distances, reachability, predecessor, processed, ordering, x,
             max_eps},
            order_idx < n_samples do
        unprocessed_mask = processed == 0
        point = Nx.argmin(Nx.select(unprocessed_mask, reachability, Nx.Constants.infinity()))
        processed = Nx.put_slice(processed, [point], Nx.new_axis(1, 0))
        ordering = Nx.put_slice(ordering, [order_idx], Nx.new_axis(point, 0))

        {reachability, predecessor} =
          set_reach_dist(core_distances, reachability, predecessor, point, processed, x,
            max_eps: max_eps
          )

        {order_idx + 1, core_distances, reachability, predecessor, processed, ordering, x,
         max_eps}
      end

    reachability =
      Nx.select(
        reachability == Nx.Constants.max_finite({:f, 32}),
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
          opts \\ []
        ) do
    max_eps = opts[:max_eps]
    n_features = Nx.axis_size(x, 1)
    n_samples = Nx.axis_size(x, 0)
    nbrs = Scholar.Neighbors.BruteKNN.fit(x, num_neighbors: n_samples)
    t = Nx.take(x, point_index, axis: 0)
    p = Nx.broadcast(t, {1, n_features})
    {neighbors, distances} = Scholar.Neighbors.BruteKNN.predict(nbrs, p)

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
    reversed_improved = Nx.max(Nx.multiply(improved, -1), 0)

    reachability =
      Nx.select(improved <= 0, Nx.take(reachability, reversed_improved), rdists)

    predecessor =
      Nx.select(improved <= 0, Nx.take(predecessor, reversed_improved), point_index)

    {reachability, predecessor}
  end

  defnp cluster_optics_dbscan(reachability, core_distances, ordering, opts \\ []) do
    eps = opts[:eps]
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
