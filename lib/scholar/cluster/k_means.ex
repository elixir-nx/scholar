defmodule Scholar.Cluster.KMeans do
  @moduledoc """
  K-Means Algorithm

  K-Means is simple clustering method that works iteratively [1]. In the first iteration,
  centroids are chosen randomly from input data. It turned out that some initialization
  are especially effective. In 2007 David Arthur and Sergei Vassilvitskii proposed initialization
  called k-means++ which speed up conergence of algorithm drastically [2]. After initialization, from each centroid
  find points that are the clostest to that centroid. Then, for each centroid replace it with the
  center of mass of associated points. These two steps mentioned above are repeated until the solution
  converge. Since some initializations are unfortunate and converge to sub-optimal results
  we need repeat the whole procedure a few times and take the best result.

  Reference:

  * [1] - [K-Means Algorithm](https://cs.nyu.edu/~roweis/csc2515-2006/readings/lloyd57.pdf)
  * [2] - [K-Means++ Initialization](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)
  """
  import Nx.Defn
  import Scholar.Shared
  require Nx

  @derive {Nx.Container, containers: [:num_iterations, :clusters, :inertia, :labels]}
  defstruct [:num_iterations, :clusters, :inertia, :labels]

  opts = [
    num_clusters: [
      required: true,
      type: :pos_integer,
      doc: "The number of clusters to form as well as the number of centroids to generate."
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
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-4,
      doc: """
      Relative tolerance with regards to Frobenius norm of the difference in
      the cluster centers of two consecutive iterations to declare convergence.
      """
    ],
    weights: [
      type: {:custom, Scholar.Options, :weights, []},
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
        on an empirical probability distribution of the points' contribution to
        the overall inertia. This technique speeds up convergence, and is
        theoretically proven to be O(log(k))-optimal.

      * `:random` - choose `:num_clusters` observations (rows) at random from data for
        the initial centroids.
      """
    ],
    seed: [
      type: :integer,
      doc: """
      Determines random number generation for centroid initialization.
      If the seed is not provided, it is set to `System.system_time()`.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a K-Means model for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:clusters` - Coordinates of cluster centers.

    * `:num_iterations` - Number of iterations run.

    * `:inertia` - Sum of squared distances of samples to their closest cluster center.

    * `:labels` - Labels of each point.

  ## Examples

      iex>  Scholar.Cluster.KMeans.fit(Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]]),
      ...>    num_clusters: 2
      ...>  )
      %Scholar.Cluster.KMeans{
        num_iterations: #Nx.Tensor<
          s64
          2
        >,
        clusters: #Nx.Tensor<
          f32[2][2]
          [
            [1.0, 2.5],
            [2.0, 4.5]
          ]
        >,
        inertia: #Nx.Tensor<
          f32
          1.0
        >,
        labels: #Nx.Tensor<
          s64[4]
          [0, 1, 0, 1]
        >
      }
  """
  deftransform fit(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected input tensor to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    {num_samples, _num_features} = Nx.shape(x)
    opts = NimbleOptions.validate!(opts, @opts_schema)

    unless opts[:num_clusters] <= num_samples do
      raise ArgumentError,
            "invalid value for :num_clusters option: expected positive integer between 1 and #{inspect(num_samples)}, got: #{inspect(opts[:num_clusters])}"
    end

    seed = Keyword.get_lazy(opts, :seed, &System.system_time/0)
    {weights, opts} = Keyword.pop(opts, :weights, nil)
    weights = validate_weights(weights, num_samples)

    fit_n(x, weights, seed, opts)
  end

  defnp fit_n(%Nx.Tensor{shape: {num_samples, num_features}} = x, weights, seed, opts) do
    num_clusters = opts[:num_clusters]
    num_runs = opts[:num_runs]

    broadcast_weights =
      weights
      |> Nx.as_type(:f32)
      |> Nx.broadcast({num_runs, num_samples})
      |> Nx.reshape({num_runs, 1, num_samples})
      |> Nx.broadcast({num_runs, num_clusters, num_samples})

    broadcast_x =
      x
      |> Nx.broadcast({num_runs, num_samples, num_features})

    centroids = initialize_centroids(x, seed, opts)
    inf = Nx.Constants.infinity(Nx.type(centroids))
    distance = Nx.broadcast(inf, {num_runs})
    tol = (x |> Nx.variance(axes: [0]) |> Nx.mean()) * opts[:tol]

    {i, _, _, _, _, _, _, final_centroids, nearest_centroids} =
      while {i = 0, tol, x, distance, weights, broadcast_weights, broadcast_x, centroids,
             _nearest_centroids = Nx.broadcast(-1, {num_runs, num_samples})},
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

        distance =
          Scholar.Metrics.Distance.squared_euclidean(centroids, previous_iteration_centroids,
            axes: [1, 2]
          )

        {i + 1, tol, x, distance, weights, broadcast_weights, broadcast_x, centroids,
         nearest_centroids}
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

  defnp initialize_centroids(x, seed, opts) do
    num_clusters = opts[:num_clusters]
    {num_samples, _num_features} = Nx.shape(x)
    x = to_float(x)
    num_runs = opts[:num_runs]

    case opts[:init] do
      :random ->
        key = Nx.Random.key(seed)
        nums = Nx.iota({num_runs, num_samples}, axis: 1)
        {temp, _} = Nx.Random.shuffle(key, nums, axis: 1)

        temp
        |> Nx.slice_along_axis(0, num_clusters, axis: 1)
        |> then(&Nx.take(x, &1))

      :k_means_plus_plus ->
        k_means_plus_plus(x, num_clusters, num_runs, seed)
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

  defnp find_new_centroid(weights, x, num_runs, random_key) do
    {rand, new_key} = Nx.Random.uniform(random_key, shape: {num_runs}, type: :f32)
    val = (rand * Nx.sum(weights, axes: [1])) |> Nx.new_axis(-1)
    cumulative_weights = Nx.cumulative_sum(weights, axis: 1)

    idx =
      (val <= cumulative_weights)
      |> Nx.as_type({:s, 8})
      |> Nx.argmax(tie_break: :low, axis: 1)

    {Nx.take(x, idx), new_key}
  end

  defnp k_means_plus_plus(x, num_clusters, num_runs, seed) do
    inf = Nx.Constants.infinity()
    {num_samples, num_features} = Nx.shape(x)
    centroids = Nx.broadcast(inf, {num_runs, num_clusters, num_features})
    inertia = Nx.broadcast(0.0, {num_runs, num_samples})

    key = Nx.Random.key(seed)

    {first_centroid_idx, new_key} = Nx.Random.randint(key, 0, num_samples - 1, shape: {num_runs})
    first_centroid = Nx.take(x, first_centroid_idx)
    centroids = Nx.put_slice(centroids, [0, 0, 0], Nx.new_axis(first_centroid, 1))

    {_, _, _, _, final_centroids} =
      while {idx = 1, x, inertia, random_key = new_key, centroids}, idx < num_clusters do
        {_inertia_for_centroids, min_inertia} =
          calculate_inertia(x, centroids, num_clusters, num_runs)

        {new_centroid, new_key} = find_new_centroid(min_inertia, x, num_runs, random_key)
        centroids = Nx.put_slice(centroids, [0, idx, 0], Nx.new_axis(new_centroid, 1))
        {idx + 1, x, inertia, new_key, centroids}
      end

    final_centroids
  end

  @doc """
  Makes predictions with the given `model` on inputs `x`.

  ## Return Values

    It returns a tensor with clusters corresponding to the input.

  ## Examples

      iex> model =
      ...>  Scholar.Cluster.KMeans.fit(Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]]),
      ...>    num_clusters: 2
      ...>  )
      iex> Scholar.Cluster.KMeans.predict(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      #Nx.Tensor<
        s64[2]
        [1, 0]
      >
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
  Calculates distances between each sample from `x` and the calculated centroids.

  ## Return Values

    It returns a tensor with corresponding distances.

  ## Examples

      iex> model =
      ...>  Scholar.Cluster.KMeans.fit(Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]]),
      ...>    num_clusters: 2
      ...>  )
      iex> Scholar.Cluster.KMeans.transform(model, Nx.tensor([[1.0, 2.5]]))
      #Nx.Tensor<
        f32[1][2]
        [
          [2.2360680103302, 0.0]
        ]
      >
  """
  defn transform(%__MODULE__{clusters: clusters} = _model, x) do
    {num_clusters, num_features} = Nx.shape(clusters)
    {num_samples, _} = Nx.shape(x)

    Scholar.Metrics.Distance.euclidean(
      Nx.new_axis(x, 1) |> Nx.broadcast({num_samples, num_clusters, num_features}),
      Nx.new_axis(clusters, 0) |> Nx.broadcast({num_samples, num_clusters, num_features}),
      axes: [-1]
    )
  end

  deftransformp validate_weights(weights, num_samples) do
    cond do
      is_nil(weights) ->
        1.0

      Nx.is_tensor(weights) and Nx.shape(weights) == {num_samples} ->
        weights

      is_list(weights) and length(weights) == num_samples ->
        Nx.tensor(weights, type: :f32)

      true ->
        raise ArgumentError,
              "invalid value for :weights option: expected list or tensor of positive numbers of size #{num_samples}, got: #{inspect(weights)}"
    end
  end
end
