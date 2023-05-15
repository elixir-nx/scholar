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
  Fits a PCA for sample inputs `x`.

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
          [1428.49609375, 0.0],
          [-1499.8349609375, 0.0],
          [-418.6555480957031, 0.0],
          [490.010009765625, 0.0]
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
      {opts[:perplexity], opts[:learning_rate], opts[:num_iters],
       opts[:num_components], opts[:exaggeration], opts[:init], opts[:metric]}

    x = to_float(x)
    {n, _dims} = Nx.shape(x)

    y = case init do
      :random ->
        key = Nx.Random.key(seed)
        {y, _new_key} = Nx.Random.normal(key, 0.0, 1.0e-4, shape: {n, num_components}, type: Nx.type(x))
        y
      :pca ->
        Scholar.Decomposition.PCA.fit_transform(x, num_components: num_components)
    end
    ys = Nx.broadcast(0.0, {num_iters, n, num_components})

    ys = Nx.put_slice(ys, [0, 0, 0], Nx.new_axis(y, 0))
    ys = Nx.put_slice(ys, [1, 0, 0], Nx.new_axis(y, 0))

    p = p_joint(x, perplexity, metric)

    {y, _, _, _, _} =
      while {_y = y, ys, learning_rate, p, i = 2}, i < num_iters do
        q = q_joint(Nx.take(ys, i - 1), metric)
        grad = gradient(p*exaggeration(i, exaggeration), q, Nx.take(ys, i - 1), metric)

        temp =
          Nx.take(ys, i - 1) - learning_rate * grad+
            +momentum(i) * (Nx.take(ys, i - 1) - Nx.take(ys, i - 2))

        ys = Nx.put_slice(ys, [i, 0, 0], Nx.new_axis(temp, 0))
        {temp, ys, learning_rate, p, i + 1}
      end

    y
  end

  defnp pairwise_dist(x, metric) do
    {num_samples, num_features} = Nx.shape(x)

    t1 = Nx.new_axis(x, 0) |> Nx.broadcast({num_samples, num_samples, num_features})
    t2 = Nx.new_axis(x, 1) |> Nx.broadcast({num_samples, num_samples, num_features})

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
    p = Nx.exp(-distances / (2 * Nx.reshape(sigmas, {:auto, 1})) ** 2)
    {n, _} = Nx.shape(p)
    p = Nx.put_diagonal(p, Nx.broadcast(0, {n}))
    p = p + Nx.Constants.epsilon(:f32)
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
    inv_distances = Nx.put_diagonal(inv_distances, Nx.broadcast(0, {n}))
    inv_distances / Nx.sum(inv_distances)
  end

  defnp gradient(p, q, y, metric) do
    pq_diff = p - q
    y_diff = Nx.new_axis(y, 1) - Nx.new_axis(y, 0)
    distances = pairwise_dist(y, metric)

    inv_distances = 1 / (1 + distances)

    ((4 * (Nx.new_axis(pq_diff, 2) * y_diff * Nx.new_axis(inv_distances, 2)))
     |> Nx.sum(axes: [1])) * 2
  end

  defnp momentum(t) do
    if t < 250 do
      0.5
    else
      0.8
    end
  end

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
