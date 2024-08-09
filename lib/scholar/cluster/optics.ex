defmodule Scholar.Cluster.OPTICS do
  @moduledoc """
  OPTICS (Ordering Points To Identify the Clustering Structure), closely related to DBSCAN, finds core sample of high density and expands clusters from them. Unlike DBSCAN, keeps cluster hierarchy for a variable neighborhood radius. Clusters are then extracted using a DBSCAN-like method.
  """
  import Nx.Defn
  require Nx

  opts = [
    min_samples: [
      default: 5,
      type: :pos_integer,
      doc: "The number of samples in a neighborhood for a point to be considered as a core point."
    ],
    max_eps: [
      default: Nx.Constants.infinity(),
      type: {:custom, Scholar.Options, :beta, []},
      doc:
        "The maximum distance between two samples for one to be considered as in the neighborhood of the other. Default value of Nx.Constants.infinity() will identify clusters across all scales "
    ],
    eps: [
      default: Nx.Constants.nan(),
      type: {:custom, Scholar.Options, :beta, []},
      doc:
        "The maximum distance between two samples for one to be considered as in the neighborhood of the other. By default it assumes the same value as max_eps."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """

  Perform OPTICS clustering for `x` which is tensor of {n_samples, n_features} shape.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a labels tensor of shape {n_samples}
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
  """

  deftransform fit(x, opts \\ []) do
    fit_p(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_p(x, opts \\ []) do
    {core_distances, reachability, _predecessor, ordering} = compute_optics_graph(x, opts)
    eps = opts[:eps]
    cluster_optics_dbscan(reachability, core_distances, ordering, eps: eps)
  end

  defnp compute_optics_graph(x, opts \\ []) do
    min_samples = opts[:min_samples]
    max_eps = opts[:max_eps]
    n_samples = Nx.axis_size(x, 0)
    reachability = Nx.broadcast(Nx.Constants.max_finite({:f, 32}), {n_samples})
    predecessor = Nx.broadcast(-1, {n_samples})
    neighbors = Scholar.Neighbors.BruteKNN.fit(x, num_neighbors: min_samples)
    core_distances = compute_core_distances(x, neighbors, min_samples: min_samples)

    core_distances =
      Nx.select(Nx.greater(core_distances, max_eps), Nx.Constants.infinity(), core_distances)

    ordering = Nx.broadcast(0, {n_samples})
    processed = Nx.broadcast(0, {n_samples})

    {_order_idx, core_distances, reachability, predecessor, _processed, ordering, _x, _max_eps} =
      while {order_idx = 0, core_distances, reachability, predecessor, processed, ordering, x,
             max_eps},
            order_idx < n_samples do
        unprocessed_mask = Nx.equal(processed, 0)
        point = Nx.argmin(Nx.select(unprocessed_mask, reachability, Nx.Constants.infinity()))
        processed = Nx.put_slice(processed, [point], Nx.broadcast(1, {1}))
        ordering = Nx.put_slice(ordering, [order_idx], Nx.broadcast(point, {1}))

        {reachability, predecessor} =
          set_reach_dist(core_distances, reachability, predecessor, point, processed, x,
            max_eps: max_eps
          )

        {order_idx + 1, core_distances, reachability, predecessor, processed, ordering, x,
         max_eps}
      end

    reachability =
      Nx.select(
        Nx.equal(reachability, Nx.Constants.max_finite({:f, 32})),
        Nx.Constants.infinity(),
        reachability
      )

    {core_distances, reachability, predecessor, ordering}
  end

  defnp compute_core_distances(x, neighbors, opts \\ []) do
    {_neighbors, distances} = Scholar.Neighbors.BruteKNN.predict(neighbors, x)
    Nx.slice_along_axis(distances, opts[:min_samples] - 1, 1, axis: 1)
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
        Nx.logical_or(are_neighbors_processed, Nx.greater(distances, max_eps)),
        Nx.multiply(-1, neighbors),
        neighbors
      )

    dists = Nx.flatten(Scholar.Metrics.Distance.pairwise_minkowski(p, x))
    core_distance = Nx.take(core_distances, point_index)
    rdists = Nx.max(dists, core_distance)
    improved = Nx.less(rdists, reachability)
    improved = Nx.select(improved, filtered_neighbors, -1)

    improved =
      Nx.select(
        Nx.logical_and(Nx.equal(improved, -1), Nx.greater(filtered_neighbors, 0)),
        Nx.multiply(filtered_neighbors, -1),
        filtered_neighbors
      )

    rdists = Nx.select(Nx.greater_equal(improved, 0), rdists, 0)
    reversed_improved = Nx.max(Nx.multiply(improved, -1), 0)

    reachability =
      Nx.select(Nx.less_equal(improved, 0), Nx.take(reachability, reversed_improved), rdists)

    predecessor =
      Nx.select(Nx.less_equal(improved, 0), Nx.take(predecessor, reversed_improved), point_index)

    {reachability, predecessor}
  end

  defnp cluster_optics_dbscan(reachability, core_distances, ordering, opts \\ []) do
    eps = opts[:eps]
    far_reach = Nx.flatten(Nx.greater(reachability, eps))
    near_core = Nx.flatten(Nx.less_equal(core_distances, eps))
    far_and_not_near = Nx.multiply(far_reach, Nx.subtract(1, near_core))
    far_reach = Nx.take(far_reach, ordering)
    near_core = Nx.take(near_core, ordering)
    far_and_near = Nx.multiply(far_reach, near_core)
    Nx.as_type(Nx.cumulative_sum(far_and_near), :s8)
    labels = Nx.subtract(Nx.as_type(Nx.cumulative_sum(far_and_near), :s8), 1)
    labels = Nx.take(labels, Nx.argsort(ordering))
    Nx.select(far_and_not_near, -1, labels)
  end
end
