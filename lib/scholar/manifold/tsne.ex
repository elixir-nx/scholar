defmodule Scholar.Manifold.TSNE do
  @moduledoc """
  t-SNE (t-Distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction technique.

  This is an exact implementation of t-SNE and therefore it has time complexity is $O(N^2)$ for $N$ samples.

  ## Reference

  * [Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(11).](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
  """
  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Metrics.Distance

  opts_schema = [
    num_components: [
      type: :pos_integer,
      default: 2,
      doc: ~S"""
      Dimension of the embedded space.
      """
    ],
    perplexity: [
      type: :pos_integer,
      default: 30,
      doc: ~S"""
      The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
      Larger datasets usually require a larger perplexity.
      Consider selecting a value between 5 and 50.
      """
    ],
    learning_rate: [
      type: {:or, [:pos_integer, :float]},
      default: 500,
      doc: ~S"""
      The learning rate for t-SNE is usually in the range [10.0, 1000.0].
      If the learning rate is too high, the data may look like a 'ball' with any point approximately equidistant from its nearest neighbors.
      If the learning rate is too low, most points may look compressed in a dense cloud with few outliers.
      If the cost function gets stuck in a bad local minimum increasing the learning rate may help.
      """
    ],
    num_iters: [
      type: :pos_integer,
      default: 500,
      doc: ~S"""
      Maximum number of iterations for the optimization.
      """
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Determines random number generation for centroid initialization.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ],
    init: [
      type: {:in, [:random, :pca]},
      default: :pca,
      doc: ~S"""
      Initialization of embedding.
      Possible options are `:random` and `:pca`.
      """
    ],
    metric: [
      type: {:in, [:euclidean, :squared_euclidean, :cosine, :manhattan, :chebyshev]},
      default: :squared_euclidean,
      doc: ~S"""
      Metric used to compute the distances.
      """
    ],
    exaggeration: [
      type: {:or, [:float, :pos_integer]},
      default: 10.0,
      doc: ~S"""
      Controls how tight natural clusters in the original space are in the embedded space and
      how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space.
      """
    ],
    learning_loop_unroll: [
      type: :boolean,
      default: false,
      doc: ~S"""
      If `true`, the learning loop is unrolled.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Fits tSNE for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  Returns the embedded data `y`.

  ## Examples

      iex> x = Nx.iota({4,5})
      iex> key = Nx.Random.key(42)
      iex> Scholar.Manifold.TSNE.fit(x, num_components: 2, key: key)
      #Nx.Tensor<
        f32[4][2]
        [
          [-2197.154296875, 0.0],
          [-1055.148681640625, 0.0],
          [1055.148681640625, 0.0],
          [2197.154296875, 0.0]
        ]
      >
  """
  deftransform fit(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_n(x, key, opts)
  end

  defnp fit_n(x, key, opts \\ []) do
    {perplexity, learning_rate, num_iters, num_components, exaggeration, init} =
      {opts[:perplexity], opts[:learning_rate], opts[:num_iters], opts[:num_components],
       opts[:exaggeration], opts[:init]}

    x = to_float(x)
    {n, _dims} = Nx.shape(x)

    y1 =
      case init do
        :random ->
          {y, _new_key} =
            Nx.Random.normal(key, 0.0, 1.0e-4, shape: {n, num_components}, type: Nx.type(x))

          y

        :pca ->
          x_embedded = Scholar.Decomposition.PCA.fit_transform(x, num_components: num_components)
          x_embedded / Nx.standard_deviation(x_embedded[[.., 0]]) * 1.0e-4
      end

    p = p_joint(x, perplexity, opts)

    {y, _} =
      while {y1, {y2 = y1, learning_rate, p}},
            i <- 2..(num_iters - 1),
            unroll: opts[:learning_loop_unroll] do
        q = q_joint(y1, opts)
        grad = gradient(p * exaggeration(i, exaggeration), q, y1, opts)
        y_next = y1 - learning_rate * grad + momentum(i) * (y1 - y2)
        {y_next, {y1, learning_rate, p}}
      end

    y
  end

  defn pairwise_dist(x, opts) do
    {num_samples, num_features} = Nx.shape(x)
    broadcast_shape = {num_samples, num_samples, num_features}

    t1 =
      x
      |> Nx.reshape({1, num_samples, num_features})
      |> Nx.broadcast(broadcast_shape)

    t2 =
      x
      |> Nx.reshape({num_samples, 1, num_features})
      |> Nx.broadcast(broadcast_shape)

    case opts[:metric] do
      :squared_euclidean ->
        Distance.pairwise_squared_euclidean(x)

      :euclidean ->
        Distance.pairwise_euclidean(x)

      :manhattan ->
        Distance.manhattan(t1, t2, axes: [2])

      :cosine ->
        Distance.pairwise_cosine(x)

      :chebyshev ->
        Distance.chebyshev(t1, t2, axes: [2])
    end
  end

  defnp p_conditional(distances, sigmas) do
    arg = -distances / (2 * Nx.reshape(sigmas, {:auto, 1})) ** 2
    {n, _} = Nx.shape(arg)

    # Set diagonal to a large negative number so it becomes 0 after applying exp()
    min_value = Nx.Constants.min_finite(Nx.type(arg))
    arg_with_min_diag = Nx.put_diagonal(arg, Nx.broadcast(min_value, {n}))

    stabilization_constant = Nx.reduce_max(arg_with_min_diag, axes: [1], keep_axes: true)
    arg = arg - stabilization_constant

    p = Nx.exp(arg)
    p / Nx.sum(p, axes: [1], keep_axes: true)
  end

  defnp perplexity(p_matrix) do
    # Nx.select is used below so that if the entry is eps or less, we treat it as 0,
    # and this makes it so we can avoid 0 * -inf == nan issues
    eps = Nx.Constants.epsilon(Nx.type(p_matrix))
    shannon_entropy_partials = Nx.select(p_matrix <= eps, 0, p_matrix * Nx.log2(p_matrix))
    shannon_entropy = -Nx.sum(shannon_entropy_partials, axes: [1])
    2 ** shannon_entropy
  end

  defnp find_sigmas(distances, target_perplexity) do
    {n, _} = Nx.shape(distances)
    sigmas = Nx.broadcast(Nx.tensor(0, type: Nx.type(distances)), {n})

    {sigmas, _, _, _} =
      while {sigmas = sigmas, distances = distances, target_perplexity, i = 0}, i < n do
        distances_i = Nx.take(distances, i)

        sigmas =
          Nx.indexed_put(
            sigmas,
            Nx.reshape(i, {1, 1}),
            Nx.new_axis(binary_search(distances_i, target_perplexity), -1)
          )

        {sigmas, distances, target_perplexity, i + 1}
      end

    sigmas
  end

  defnp binary_search(distances, target_perplexity, opts \\ []) do
    opts = keyword!(opts, tol: 1.0e-5, max_iters: 100, low: 1.0e-20, high: 1.0e4)
    {low, high, max_iters, tol} = {opts[:low], opts[:high], opts[:max_iters], opts[:tol]}

    {low, high, _} =
      while {
              low = low,
              high = high,
              {max_iters, tol,
               perplexity_val =
                 Nx.Constants.infinity(
                   Nx.Type.to_floating(
                     Nx.Type.merge(Nx.type(target_perplexity), Nx.type(distances))
                   )
                 ), distances, target_perplexity, i = 0}
            },
            i < max_iters and Nx.abs(perplexity_val - target_perplexity) > tol do
        mid = (low + high) / 2

        condition_matrix = p_conditional(distances, Nx.new_axis(mid, 0))
        perplexity_val = perplexity(condition_matrix) |> Nx.reshape({})

        {low, high} =
          if perplexity_val > target_perplexity do
            {low, mid}
          else
            {mid, high}
          end

        {low, high, {max_iters, tol, perplexity_val, distances, target_perplexity, i + 1}}
      end

    (high + low) / 2
  end

  defnp q_joint(y, opts) do
    distances = pairwise_dist(y, opts)
    n = Nx.axis_size(distances, 0)
    inv_distances = 1 / (1 + distances)
    inv_distances = inv_distances / Nx.sum(inv_distances)
    Nx.put_diagonal(inv_distances, Nx.broadcast(0, {n}))
  end

  defnp gradient(p, q, y, opts) do
    pq_diff = Nx.new_axis(p - q, 2)
    y_diff = Nx.new_axis(y, 1) - Nx.new_axis(y, 0)
    distances = pairwise_dist(y, opts)

    inv_distances = Nx.new_axis(1 / (1 + distances), 2)

    grads = 4 * (pq_diff * y_diff * inv_distances)
    Nx.sum(grads, axes: [1])
  end

  defnp momentum(t) do
    if t < 250 do
      0.5
    else
      0.8
    end
  end

  # for early exaggeration we decided to use 250 iterations
  defnp exaggeration(t, exaggeration) do
    if t < 250 do
      exaggeration
    else
      1.0
    end
  end

  defnp p_joint(x, perplexity, opts) do
    {n, _} = Nx.shape(x)
    distances = pairwise_dist(x, opts)
    sigmas = find_sigmas(distances, perplexity)
    p_cond = p_conditional(distances, sigmas)
    (p_cond + Nx.transpose(p_cond)) / (2 * n)
  end
end
