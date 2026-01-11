defmodule Scholar.Cluster.KMeans do
  @moduledoc """
  K-Means Algorithm.

  K-Means is a simple clustering method that works iteratively [1]. In the first iteration,
  centroids are chosen randomly from input data. It turned out that some initializations
  are especially effective. In 2007 David Arthur and Sergei Vassilvitskii proposed initialization
  called k-means++ which speed up convergence of algorithm drastically [2]. After initialization, from each centroid
  find points that are the closest to that centroid. Then, for each centroid replace it with the
  center of mass of associated points. These two steps mentioned above are repeated until the solution
  converges. Since some initializations are unfortunate and converge to sub-optimal results
  we need repeat the whole procedure a few times and take the best result.

  Average time complexity is $O(CKNI)$, where $C$ is the number of clusters, $N$ is the number of samples,
  $I$ is the number of iterations until convergence, and $K$ is the number of features. Space
  complexity is $O(K*(N+C))$.

  Reference:

  * [1] - [K-Means Algorithm](https://cs.nyu.edu/~roweis/csc2515-2006/readings/lloyd57.pdf)
  * [2] - [K-Means++ Initialization](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)
  """
  import Nx.Defn
  import Scholar.Shared

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
      The weights for each observation in `x`. If equals to `nil`,
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
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Determines random number generation for centroid initialization.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
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

      iex> key = Nx.Random.key(42)
      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> Scholar.Cluster.KMeans.fit(x, num_clusters: 2, key: key)
      %Scholar.Cluster.KMeans{
        num_iterations: Nx.tensor(
          2
        ),
        clusters: Nx.tensor(
          [
            [1.0, 2.5],
            [2.0, 4.5]
          ]
        ),
        inertia: Nx.tensor(
          1.0
        ),
        labels: Nx.tensor(
          [0, 1, 0, 1]
        )
      }
  """
  deftransform fit(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected input tensor to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    num_samples = Nx.axis_size(x, 0)
    opts = NimbleOptions.validate!(opts, @opts_schema)

    unless opts[:num_clusters] <= num_samples do
      raise ArgumentError,
            "invalid value for :num_clusters option: expected positive integer between 1 and #{inspect(num_samples)}, got: #{inspect(opts[:num_clusters])}"
    end

    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    {weights, opts} = Keyword.pop(opts, :weights, nil)
    weights = validate_weights(weights, num_samples, type: to_float_type(x))
    fit_n(x, weights, key, opts)
  end

  defnp fit_n(%Nx.Tensor{shape: {num_samples, num_features}} = x, weights, key, opts) do
    x = to_float(x)
    num_clusters = opts[:num_clusters]
    num_runs = opts[:num_runs]

    broadcast_weights =
      weights
      |> Nx.broadcast({num_runs, num_clusters, num_samples})

    broadcast_x = Nx.broadcast(x, {num_runs, num_samples, num_features})

    centroids = initialize_centroids(x, key, opts)
    inf = Nx.Constants.infinity(Nx.type(centroids))
    distance = Nx.broadcast(inf, {num_runs})
    tol = Nx.mean(Nx.variance(x, axes: [0])) * opts[:tol]

    {{i, final_centroids, nearest_centroids}, _} =
      while {{i = 0, centroids, _nearest_centroids = Nx.broadcast(-1, {num_runs, num_samples})},
             {tol, x, distance, weights, broadcast_weights, broadcast_x}},
            i < opts[:max_iterations] and
              Nx.all(distance > tol) do
        previous_iteration_centroids = centroids

        {inertia_for_centroids, _min_inertia} =
          calculate_inertia(x, centroids, num_clusters, num_runs)

        nearest_centroids = Nx.argmin(inertia_for_centroids, axis: 1)

        group_masks =
          (Nx.iota({num_runs, num_clusters, 1}, axis: 1) == Nx.new_axis(nearest_centroids, 1)) *
            broadcast_weights

        group_sizes = Nx.sum(group_masks, axes: [2], keep_axes: true)

        centroids =
          ((Nx.new_axis(group_masks, -1) * Nx.new_axis(broadcast_x, 1)) |> Nx.sum(axes: [2])) /
            group_sizes

        distance =
          Scholar.Metrics.Distance.squared_euclidean(centroids, previous_iteration_centroids,
            axes: [1, 2]
          )

        {{i + 1, centroids, nearest_centroids},
         {tol, x, distance, weights, broadcast_weights, broadcast_x}}
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

  defnp initialize_centroids(x, key, opts) do
    num_clusters = opts[:num_clusters]
    {num_samples, _num_features} = Nx.shape(x)
    num_runs = opts[:num_runs]

    case opts[:init] do
      :random ->
        nums = Nx.iota({num_runs, num_samples}, axis: 1)
        {temp, _} = Nx.Random.shuffle(key, nums, axis: 1)

        temp
        |> Nx.slice_along_axis(0, num_clusters, axis: 1)
        |> then(&Nx.take(x, &1))

      :k_means_plus_plus ->
        k_means_plus_plus(x, num_clusters, num_runs, key)
    end
  end

  defnp calculate_inertia(x, centroids, num_clusters, num_runs) do
    {num_samples, num_features} = Nx.shape(x)

    modified_centroids =
      centroids
      |> Nx.new_axis(2)
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

  defnp k_means_plus_plus(x, num_clusters, num_runs, key) do
    inf = Nx.Constants.infinity(to_float_type(x))
    {num_samples, num_features} = Nx.shape(x)
    centroids = Nx.broadcast(inf, {num_runs, num_clusters, num_features})
    inertia = Nx.broadcast(Nx.tensor(0.0, type: to_float_type(x)), {num_runs, num_samples})

    {first_centroid_idx, new_key} = Nx.Random.randint(key, 0, num_samples - 1, shape: {num_runs})
    first_centroid = Nx.take(x, first_centroid_idx)
    centroids = Nx.put_slice(centroids, [0, 0, 0], Nx.new_axis(first_centroid, 1))

    {final_centroids, _} =
      while {centroids, {idx = 1, x, inertia, random_key = new_key}}, idx < num_clusters do
        {_inertia_for_centroids, min_inertia} =
          calculate_inertia(x, centroids, num_clusters, num_runs)

        {new_centroid, new_key} = find_new_centroid(min_inertia, x, num_runs, random_key)
        centroids = Nx.put_slice(centroids, [0, idx, 0], Nx.new_axis(new_centroid, 1))
        {centroids, {idx + 1, x, inertia, new_key}}
      end

    final_centroids
  end

  @doc """
  Makes predictions with the given `model` on inputs `x`.

  ## Return Values

    It returns a tensor with clusters corresponding to the input.

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> model = Scholar.Cluster.KMeans.fit(x, num_clusters: 2, key: key)
      iex> Scholar.Cluster.KMeans.predict(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      Nx.tensor(
        [1, 0]
      )
  """
  defn predict(%__MODULE__{clusters: clusters} = _model, x) do
    assert_same_shape!(x[0], clusters[0])

    Scholar.Metrics.Distance.pairwise_squared_euclidean(clusters, x) |> Nx.argmin(axis: 0)
  end

  @doc """
  Calculates distances between each sample from `x` and the calculated centroids.

  ## Return Values

    It returns a tensor with corresponding distances.

  ## Examples

      iex> key = Nx.Random.key(40)
      iex> model =
      ...>  Scholar.Cluster.KMeans.fit(Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]]),
      ...>    num_clusters: 2,
      ...>    key: key
      ...>  )
      iex> Scholar.Cluster.KMeans.transform(model, Nx.tensor([[1.0, 2.5]]))
      Nx.tensor(
        [
          [2.2360680103302, 0.0]
        ]
      )
  """
  defn transform(%__MODULE__{clusters: clusters} = _model, x) do
    Scholar.Metrics.Distance.pairwise_euclidean(x, clusters)
  end
end
