defmodule Scholar.Cluster.KMeans do
  @moduledoc """
  K-Means algorithm.
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:num_iterations, :clusters, :inertia, :labels]}
  defstruct [:num_iterations, :clusters, :inertia, :labels]

  @doc """
  Fits a K-Means model for sample inputs `x`


  ## Options

    * `:num_clusters` - The number of clusters to form as well as the number of centroids to generate. Required.

    * `:max_iterations` - Maximum number of iterations of the k-means algorithm for a single run. Defaults to `300`.

    * `:num_runs` - Number of time the k-means algorithm will be run with different centroid seeds.
      The final results will be the best output of num_runs runs in terms of inertia. Defaults to `10`.

    * `:tol` - Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two
      consecutive iterations to declare convergence. Defaults to `1e-4`.

    * `:random_state` - Determines random number generation for centroid initialization.
      Use an int to make the randomness deterministic.
      The argument `nil` is special and means that the seed is not set. Defaults to `nil`.

    * `:weights` - The weights for each observation in x. If equals to `nil`, all observations
      are assigned equal weight.

    * `:init` - Method for centroid initialization, either of:

        * `:k_means_plus_plus` (default) -  selects initial cluster centroids using sampling based on
          an empirical probability distribution of the points’ contribution to the overall inertia.
         This technique speeds up convergence, and is theoretically proven to be O(log(k))-optimal.

        * `:random` - choose `:num_clusters` observations (rows) at random from data for the initial centroids.

  ## Returns

    The function returns a struct with the following parameters:

    * `:clusters` - Coordinates of cluster centers.

    * `:num_iterations` - Number of iterations run.

    * `:inertia` - Sum of squared distances of samples to their closest cluster center.

    * `:labels` - Labels of each point.
  """

  defn fit(x, opts \\ []) do
    opts =
      keyword!(
        opts,
        [
          :num_clusters,
          max_iterations: 300,
          num_runs: 10,
          tol: 1.0e-4,
          random_state: nil,
          init: :k_means_plus_plus,
          weights: nil
        ]
      )

    verify(x, opts)
    inf = Nx.Constants.infinity({:f, 32})
    {num_samples, num_features} = Nx.shape(x)
    num_clusters = opts[:num_clusters]
    num_runs = opts[:num_runs]

    weights = validate_weights(opts[:weights], num_samples, num_runs)

    broadcast_weights =
      weights
      |> Nx.reshape({num_runs, 1, num_samples})
      |> Nx.broadcast({num_runs, num_clusters, num_samples})

    broadcast_x =
      x
      |> Nx.broadcast({num_runs, num_samples, num_features})

    centroids = initialize_centroids(x, opts)
    distance = Nx.broadcast(inf, {num_runs})

    {i, _, _, _, _, _, _, _, final_centroids, nearest_centroids} =
      while {i = 0, x, previous_iteration_centroids = Nx.broadcast(inf, centroids), distance,
             inertia = Nx.broadcast(0.0, {num_runs}), weights, broadcast_weights, broadcast_x,
             centroids, nearest_centroids = Nx.broadcast(-1, {num_runs, num_samples})},
            i < opts[:max_iterations] and
              Nx.all(distance > opts[:tol]) do
        previous_iteration_centroids = centroids

        {inertia_for_centroids, min_inertia} =
          calculate_inertia(x, centroids, num_clusters, num_runs)

        nearest_centroids = Nx.argmin(inertia_for_centroids, axis: 1)

        inertia = (min_inertia * weights) |> Nx.sum(axes: [1])

        group_masks =
          (Nx.broadcast(Nx.iota({num_clusters, 1}), {num_runs, num_clusters, 1}) ==
             Nx.reshape(nearest_centroids, {num_runs, 1, num_samples})) * broadcast_weights

        group_sizes = Nx.sum(group_masks, axes: [2], keep_axes: true)

        centroids =
          ((Nx.reshape(group_masks, {num_runs, num_clusters, num_samples, 1}) *
              Nx.reshape(broadcast_x, {num_runs, 1, num_samples, num_features}))
           |> Nx.sum(axes: [2])) / group_sizes

        distance = Nx.sum(Nx.abs(centroids - previous_iteration_centroids), axes: [1, 2])

        {i + 1, x, previous_iteration_centroids, distance, inertia, weights, broadcast_weights,
         broadcast_x, centroids, nearest_centroids}
      end

    {_inertia_for_centroids, min_inertia} =
      calculate_inertia(x, final_centroids, num_clusters, num_runs)

    final_inertia = (min_inertia * weights) |> Nx.sum(axes: [1])
    best_run = Nx.argmin(final_inertia)

    %__MODULE__{
      num_iterations: i,
      clusters: final_centroids[best_run],
      labels: nearest_centroids[best_run],
      inertia: final_inertia[best_run]
    }
  end

  defnp initialize_centroids(x, opts) do
    num_clusters = opts[:num_clusters]
    {num_samples, _num_features} = Nx.shape(x)
    num_runs = opts[:num_runs]

    case opts[:init] do
      :random ->
        Nx.iota({num_runs, num_samples}, axis: 1)
        |> Nx.shuffle(axis: 1)
        |> Nx.slice_along_axis(0, num_clusters, axis: 1)
        |> then(&Nx.take(x, &1))

      :k_means_plus_plus ->
        k_means_plus_plus(x, num_clusters, num_runs)
    end
  end

  defnp calculate_inertia(x, centroids, num_clusters, num_runs) do
    {num_samples, num_features} = Nx.shape(x)

    modified_centroids =
      centroids
      |> Nx.reshape({num_runs, num_clusters, 1, num_features})
      |> Nx.broadcast({num_runs, num_clusters, num_samples, num_features})
      |> Nx.reshape({num_runs, num_clusters * num_samples, num_features})

    inertia_for_centroids =
      Scholar.Cluster.Utils.squared_euclidean(
        Nx.tile(x, [num_runs, num_clusters, 1]),
        modified_centroids
      )
      |> Nx.reshape({num_runs, num_clusters, num_samples})

    {inertia_for_centroids, Nx.reduce_min(inertia_for_centroids, axes: [1])}
  end

  defnp generate_idx(mod_weights, num_runs) do
    rand = Nx.random_uniform({num_runs}, 0, 1, type: {:f, 32})
    val = (rand * Nx.sum(mod_weights, axes: [1])) |> Nx.new_axis(-1)
    cumulative_weights = Nx.cumulative_sum(mod_weights, axis: 1)
    (val <= cumulative_weights) |> Nx.argmax(tie_break: :low, axis: 1)
  end

  defnp find_new_centroid(weights, x, centroid_mask, num_runs) do
    mod_weights = weights * centroid_mask
    idx = generate_idx(mod_weights, num_runs)
    {Nx.take(x, idx), idx}
  end

  defnp k_means_plus_plus(x, num_clusters, num_runs) do
    inf = Nx.Constants.infinity()
    {num_samples, num_features} = Nx.shape(x)
    centroids = Nx.broadcast(inf, {num_runs, num_clusters, num_features})
    inertia = Nx.broadcast(0.0, {num_runs, num_samples})
    centroid_mask = Nx.broadcast(1, {num_runs, num_samples})

    first_centroid_idx = Nx.random_uniform({num_runs}, 0, num_samples - 1, type: {:u, 32})

    first_centroid = Nx.take(x, first_centroid_idx) |> Nx.flatten()

    indices_centroids =
      Nx.stack(
        [
          Nx.flatten(Nx.iota({num_runs, num_features}, axis: 0)),
          Nx.broadcast(0, {num_runs * num_features}),
          Nx.tile(Nx.iota({num_features}), [num_runs])
        ],
        axis: -1
      )

    centroids = Nx.indexed_put(centroids, indices_centroids, first_centroid)

    indices_mask = Nx.stack([Nx.iota({num_runs}), first_centroid_idx], axis: -1)

    centroid_mask = Nx.indexed_put(centroid_mask, indices_mask, Nx.broadcast(0, {num_runs}))

    {_, _, _, _, final_centroids} =
      while {idx = 1, centroid_mask, x, inertia, centroids},
            Nx.less(idx, num_clusters) do
        {_inertia_for_centroids, min_inertia} =
          calculate_inertia(x, centroids, num_clusters, num_runs)

        {new_centroid, centroid_idx} = find_new_centroid(min_inertia, x, num_clusters, num_runs)

        indices_centroids =
          Nx.stack(
            [
              Nx.flatten(Nx.iota({num_runs, num_features}, axis: 0)),
              Nx.broadcast(idx, {num_runs * num_features}),
              Nx.tile(Nx.iota({num_features}), [num_runs])
            ],
            axis: -1
          )

        centroids = Nx.indexed_put(centroids, indices_centroids, Nx.flatten(new_centroid))

        indices_mask = Nx.stack([Nx.iota({num_runs}), centroid_idx], axis: -1)

        centroid_mask = Nx.indexed_put(centroid_mask, indices_mask, Nx.broadcast(0, {num_runs}))
        {idx + 1, centroid_mask, x, inertia, centroids}
      end

    final_centroids
  end

  # Function checks validity of the provided data

  deftransformp verify(x, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    {num_samples, _num_features} = Nx.shape(x)

    unless opts[:num_clusters] do
      raise ArgumentError,
            "missing option :num_clusters"
    end

    unless is_integer(opts[:num_clusters]) and opts[:num_clusters] > 0 and
             opts[:num_clusters] <= num_samples do
      raise ArgumentError,
            "expected :num_clusters to to be a positive integer in range 1 to #{inspect(num_samples)}, got: #{inspect(opts[:num_clusters])}"
    end

    unless opts[:init] in [:random, :k_means_plus_plus] do
      raise ArgumentError,
            "expected :init to be either :random or :k_means_plus_plus, got: #{inspect(opts[:init])}"
    end

    unless is_integer(opts[:max_iterations]) and opts[:max_iterations] > 0 do
      raise ArgumentError,
            "expected :max_iterations to be a positive integer, got: #{inspect(opts[:max_iterations])}"
    end

    unless is_integer(opts[:num_runs]) and opts[:num_runs] > 0 do
      raise ArgumentError,
            "expected :num_runs to be a positive integer, got: #{inspect(opts[:num_runs])}"
    end

    unless is_number(opts[:tol]) and opts[:tol] >= 0 do
      raise ArgumentError,
            "expected :tol to be a non-negative number, got: #{inspect(opts[:tol])}"
    end

    unless is_integer(opts[:random_state]) or is_nil(opts[:random_state]) do
      raise ArgumentError,
            "expected :random_state to be an integer or nil, got: #{inspect(opts[:random_state])}"
    end

    if opts[:random_state] != nil do
      :rand.seed(:exsss, opts[:random_state])
    end
  end

  @doc """
  Makes predictions with the given model on inputs `x`.

  It returns a tensor with clusters corresponding to the input.
  """

  defn predict(%__MODULE__{clusters: clusters} = _module, x) do
    assert_same_shape!(x[0], clusters[0])
    {num_clusters, _} = Nx.shape(clusters)
    {num_samples, num_features} = Nx.shape(x)

    clusters =
      clusters
      |> Nx.reshape({num_clusters, 1, num_features})
      |> Nx.broadcast({num_clusters, num_samples, num_features})
      |> Nx.reshape({num_clusters * num_samples, num_features})

    inertia_for_centroids =
      Scholar.Cluster.Utils.squared_euclidean(
        Nx.tile(x, [num_clusters, 1]),
        clusters,
        axes: 1
      )
      |> Nx.reshape({num_clusters, num_samples})

    inertia_for_centroids |> Nx.argmin(axis: 0)
  end

  deftransformp validate_weights(weights, num_samples, num_runs) do
    if is_nil(weights) or
         (is_list(weights) and length(weights) == num_samples and Enum.all?(weights, &(&1 >= 0))) do
      case weights do
        nil ->
          Nx.broadcast(1.0, {num_runs, num_samples})

        _ ->
          weights
          |> Nx.tensor(type: {:f, 32})
          |> Nx.broadcast({num_runs, num_samples})
      end
    else
      raise ArgumentError,
            "expected :weights to be a list of positive numbers of size #{num_samples}, got: #{inspect(weights)}"
    end
  end
end
