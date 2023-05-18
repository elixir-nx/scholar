defmodule Scholar.Manifold.TSNE do
  @moduledoc """
  TSNE (t-Distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction technique.

  ## References

  * [t-SNE: t-Distributed Stochastic Neighbor Embedding](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
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
    seed: [
      type: :integer,
      doc: """
      Determines random number generation for centroid initialization.
      If the seed is not provided, it is set to `System.system_time()`.
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
      iex> seed = 42
      iex> Scholar.Manifold.TSNE.fit(x, num_components: 2, seed: seed)
      #Nx.Tensor<
        f32[4][2]
        [
          [287.8900146484375, 0.0],
          [-207.60089111328125, 0.0],
          [-948.1324462890625, 0.0],
          [867.8435668945312, 0.0]
        ]
      >
  """
  deftransform fit(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    seed = Keyword.get_lazy(opts, :seed, &System.system_time/0)
    fit_n(x, seed, opts)
  end

  defnp fit_n(x, seed, opts \\ []) do
    {perplexity, learning_rate, num_iters, num_components, exaggeration, init, metric} =
      {opts[:perplexity], opts[:learning_rate], opts[:num_iters], opts[:num_components],
       opts[:exaggeration], opts[:init], opts[:metric]}

    x = to_float(x)
    {n, _dims} = Nx.shape(x)

    y1 =
      case init do
        :random ->
          key = Nx.Random.key(seed)

          {y, _new_key} =
            Nx.Random.normal(key, 0.0, 1.0e-4, shape: {n, num_components}, type: Nx.type(x))

          y

        :pca ->
          Scholar.Decomposition.PCA.fit_transform(x, num_components: num_components)
      end

    p = p_joint(x, perplexity, metric)

    {y, _, _, _, _} =
      while {y1, y2 = y1, learning_rate, p, i = 2}, i < num_iters do
        q = q_joint(y1, metric)
        grad = gradient(p * exaggeration(i, exaggeration), q, y1, metric)
        y_next = y1 - learning_rate * grad + momentum(i) * (y1 - y2)
        {y_next, y1, learning_rate, p, i + 1}
      end

    y
  end

  defnp pairwise_dist(x, metric) do
    {num_samples, num_features} = Nx.shape(x)

    t1 =
      x
      |> Nx.reshape({1, num_samples, num_features})
      |> Nx.broadcast({num_samples, num_samples, num_features})

    t2 =
      x
      |> Nx.reshape({num_samples, 1, num_features})
      |> Nx.broadcast({num_samples, num_samples, num_features})

    case metric do
      :squared_euclidean ->
        Distance.squared_euclidean(t1, t2, axes: [2])

      :euclidean ->
        Distance.euclidean(t1, t2, axes: [2])

      :manhattan ->
        Distance.manhattan(t1, t2, axes: [2])

      :cosine ->
        Distance.cosine(t1, t2, axes: [2])

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

  defnp perplexity(condition_matrix) do
    exponent = -Nx.sum(condition_matrix * Nx.log2(condition_matrix), axes: [1])
    2 ** exponent
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

    {low, high, _, _, _, _, _, _} =
      while {
              low = low,
              high = high,
              max_iters,
              tol,
              perplexity_val = Nx.Constants.infinity(:f32),
              distances,
              target_perplexity,
              i = 0
            },
            i < max_iters and Nx.abs(perplexity_val - target_perplexity) > tol do
        mid = (low + high) / 2

        condition_matrix = p_conditional(distances, Nx.new_axis(mid, 0))
        perplexity_val = perplexity(condition_matrix) |> Nx.reshape({})

        high_to_mid? =
          cond do
            perplexity_val > target_perplexity ->
              1

            true ->
              0
          end

        low = if high_to_mid?, do: low, else: mid
        high = if high_to_mid?, do: mid, else: high

        {low, high, max_iters, tol, perplexity_val, distances, target_perplexity, i + 1}
      end

    (high + low) / 2
  end

  defnp q_joint(y, metric) do
    distances = pairwise_dist(y, metric)
    n = Nx.axis_size(distances, 0)
    inv_distances = 1 / (1 + distances)
    inv_distances = inv_distances / Nx.sum(inv_distances)
    Nx.put_diagonal(inv_distances, Nx.broadcast(0, {n}))
  end

  defnp gradient(p, q, y, metric) do
    pq_diff = Nx.new_axis(p - q, 2)
    y_diff = Nx.new_axis(y, 1) - Nx.new_axis(y, 0)
    distances = pairwise_dist(y, metric)

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

  defnp p_joint(x, perplexity, metric) do
    {n, _} = Nx.shape(x)
    distances = pairwise_dist(x, metric)
    sigmas = find_sigmas(distances, perplexity)
    p_cond = p_conditional(distances, sigmas)
    (p_cond + Nx.transpose(p_cond)) / (2 * n)
  end
end
