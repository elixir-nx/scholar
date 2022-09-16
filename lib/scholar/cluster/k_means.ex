defmodule Scholar.Cluster.KMeans do
  @moduledoc """
  K-Means algorithm.
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:num_iterations, :clusters, :inertia, :labels]}
  defstruct [:num_iterations, :clusters, :inertia, :labels]

  @opts_schema NimbleOptions.new!(
                 num_clusters: [
                   required: true,
                   type: :pos_integer,
                   doc:
                     "The number of clusters to form as well as the number of centroids to generate."
                 ],
                 max_iterations: [
                   type: :pos_integer,
                   default: 300,
                   doc: "Maximum number of iterations of the k-means algorithm for a single run."
                 ],
                 num_runs: [
                   type: :pos_integer,
                   default: 10,
                   doc: """
                   Number of time the k-means algorithm will be run with different centroid seeds.
                   The final results will be the best output of num_runs runs in terms of inertia.
                   """
                 ],
                 tol: [
                   type: {:custom, Scholar.Shared, :check_if_positive_float, [:tol]},
                   default: 1.0e-4,
                   doc: """
                   Relative tolerance with regards to Frobenius norm of the difference in
                   the cluster centers of two consecutive iterations to declare convergence.
                   """
                 ],
                 weights: [
                   type:
                     {:list,
                      {:or,
                       [
                         {:custom, Scholar.Shared, :check_if_positive_float, [:weights]},
                         :non_neg_integer
                       ]}},
                   doc: """
                   The weights for each observation in x. If equals to `nil`,
                   all observations are assigned equal weight.
                   """
                 ],
                 init: [
                   type: {:in, [:k_means_plus_plus, :random]},
                   default: :k_means_plus_plus,
                   doc: """
                   Method for centroid initialization, either of:

                   * `:k_means_plus_plus` - selects initial cluster centroids using sampling based
                   on an empirical probability distribution of the points’ contribution to
                   the overall inertia. This technique speeds up convergence, and is
                   theoretically proven to be O(log(k))-optimal.
                   * `:random` - choose `:num_clusters` observations (rows) at random from data for the initial centroids.
                   """
                 ]
               )

  @doc """
  Fits a K-Means model for sample inputs `x`.

  ## Options \n#{NimbleOptions.docs(@opts_schema)}

  ## Returns

    The function returns a struct with the following parameters:

    * `:clusters` - Coordinates of cluster centers.

    * `:num_iterations` - Number of iterations run.

    * `:inertia` - Sum of squared distances of samples to their closest cluster center.

    * `:labels` - Labels of each point.
  """
  deftransform train(x, opts \\ []) do
    ntrain(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp ntrain(x, opts \\ []) do
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
    tol = (x |> Nx.variance(axes: [0]) |> Nx.mean()) * opts[:tol]

    {i, _, _, _, _, _, _, _, final_centroids, nearest_centroids} =
      while {i = 0, tol, x, previous_iteration_centroids = Nx.broadcast(inf, centroids), distance,
             weights, broadcast_weights, broadcast_x, centroids,
             nearest_centroids = Nx.broadcast(-1, {num_runs, num_samples})},
            i < opts[:max_iterations] and
              Nx.all(distance > tol) do
        previous_iteration_centroids = centroids

        {inertia_for_centroids, _min_inertia} =
          calculate_inertia(x, centroids, num_clusters, num_runs)

        nearest_centroids = Nx.argmin(inertia_for_centroids, axis: 1)

        group_masks =
          (Nx.broadcast(Nx.iota({num_clusters, 1}), {num_runs, num_clusters, 1}) ==
             Nx.reshape(nearest_centroids, {num_runs, 1, num_samples})) * broadcast_weights

        group_sizes = Nx.sum(group_masks, axes: [2], keep_axes: true)

        centroids =
          ((Nx.reshape(group_masks, {num_runs, num_clusters, num_samples, 1}) *
              Nx.reshape(broadcast_x, {num_runs, 1, num_samples, num_features}))
           |> Nx.sum(axes: [2])) / group_sizes

        distance = Nx.sum((centroids - previous_iteration_centroids) ** 2, axes: [1, 2])

        {i + 1, tol, x, previous_iteration_centroids, distance, weights, broadcast_weights,
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
    x = to_float(x)
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
      Scholar.Metrics.Distance.squared_euclidean(
        Nx.tile(x, [num_runs, num_clusters, 1]),
        modified_centroids,
        axes: [2]
      )
      |> Nx.reshape({num_runs, num_clusters, num_samples})

    {inertia_for_centroids, Nx.reduce_min(inertia_for_centroids, axes: [1])}
  end

  defnp find_new_centroid(weights, x, num_runs) do
    rand = Nx.random_uniform({num_runs}, 0.0, 1.0, type: {:f, 32})
    val = (rand * Nx.sum(weights, axes: [1])) |> Nx.new_axis(-1)
    cumulative_weights = Nx.cumulative_sum(weights, axis: 1)

    idx =
      (val <= cumulative_weights)
      |> Nx.as_type({:s, 8})
      |> Nx.argmax(tie_break: :low, axis: 1)

    Nx.take(x, idx)
  end

  defnp k_means_plus_plus(x, num_clusters, num_runs) do
    inf = Nx.Constants.infinity()
    {num_samples, num_features} = Nx.shape(x)
    centroids = Nx.broadcast(inf, {num_runs, num_clusters, num_features})
    inertia = Nx.broadcast(0.0, {num_runs, num_samples})

    first_centroid_idx = Nx.random_uniform({num_runs}, 0, num_samples - 1)
    first_centroid = Nx.take(x, first_centroid_idx)
    centroids = Nx.put_slice(centroids, [0, 0, 0], Nx.new_axis(first_centroid, 1))

    {_, _, _, final_centroids} =
      while {idx = 1, x, inertia, centroids}, idx < num_clusters do
        {_inertia_for_centroids, min_inertia} =
          calculate_inertia(x, centroids, num_clusters, num_runs)

        new_centroid = find_new_centroid(min_inertia, x, num_runs)
        centroids = Nx.put_slice(centroids, [0, idx, 0], Nx.new_axis(new_centroid, 1))
        {idx + 1, x, inertia, centroids}
      end

    final_centroids
  end

  deftransformp verify(x, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    {num_samples, _num_features} = Nx.shape(x)

    unless opts[:num_clusters] <= num_samples do
      raise ArgumentError,
            "expected :num_clusters to to be a positive integer in range 1 to #{inspect(num_samples)}, got: #{inspect(opts[:num_clusters])}"
    end

    unless is_number(opts[:tol]) and opts[:tol] >= 0 do
      raise ArgumentError,
            "expected :tol to be a non-negative number, got: #{inspect(opts[:tol])}"
    end
  end

  @doc """
  Makes predictions with the given model on inputs `x`.

  It returns a tensor with clusters corresponding to the input.
  """

  defn predict(%__MODULE__{clusters: clusters} = _model, x) do
    assert_same_shape!(x[0], clusters[0])
    {num_clusters, _} = Nx.shape(clusters)
    {num_samples, num_features} = Nx.shape(x)

    clusters =
      clusters
      |> Nx.reshape({num_clusters, 1, num_features})
      |> Nx.broadcast({num_clusters, num_samples, num_features})
      |> Nx.reshape({num_clusters * num_samples, num_features})

    inertia_for_centroids =
      Scholar.Metrics.Distance.squared_euclidean(
        Nx.tile(x, [num_clusters, 1]),
        clusters,
        axes: [1]
      )
      |> Nx.reshape({num_clusters, num_samples})

    inertia_for_centroids |> Nx.argmin(axis: 0)
  end

  @doc """
  For model and input `x` function calculate distances between each sample from `x` and centroids
  """

  defn transform(%__MODULE__{clusters: clusters} = _model, x) do
    Nx.subtract(
      Nx.new_axis(x, 1),
      Nx.new_axis(clusters, 0)
    )
    |> Nx.power(2)
    |> Nx.sum(axes: [-1])
    |> Nx.sqrt()
  end

  deftransformp validate_weights(weights, num_samples, num_runs) do
    if is_nil(weights) or
         (is_list(weights) and length(weights) == num_samples) do
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

  deftransformp to_float(tensor) do
    type = tensor |> Nx.type() |> Nx.Type.to_floating()
    Nx.as_type(tensor, type)
  end
end
