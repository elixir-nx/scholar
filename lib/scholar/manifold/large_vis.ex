defmodule Scholar.Manifold.LargeVis do
  @moduledoc ~S"""
  LargeVis, a nonlinear dimensionality reduction technique for visualizing large
  high-dimensional datasets.

  Unlike `Scholar.Manifold.TSNE`, which computes an exact $O(N^2)$ affinity
  matrix, LargeVis builds an approximate k-nearest-neighbor graph (via
  `Scholar.Neighbors.LargeVis`) and optimizes the low-dimensional embedding with
  stochastic gradient descent over edges sampled from that graph, which scales
  to much larger datasets.

  ## Algorithm sketch

  For each point $i$ and its $k$ approximate nearest neighbors, an edge weight
  $p_{j|i}$ is computed the same way as t-SNE's conditional probabilities (a
  Gaussian kernel around $i$ with bandwidth chosen by binary search to match
  the target `:perplexity`), but only over the $k$ neighbors instead of every
  other point.

  Symmetrizing this directed k-NN graph would normally require the *reverse*
  graph, whose in-degree per node is unbounded and does not fit a fixed-shape
  tensor. This implementation avoids ever materializing that reverse graph: the
  k-NN graph is treated as a fixed-size edge list `{num_samples * num_neighbors}`
  of `(source, target, weight)` triples, and its reverse is simply that same
  list with the source and target columns swapped, still fixed-size.
  Concatenating the two lists is exactly equivalent to summing the dense
  $P + P^T$ matrix, without ever forming it.

  The embedding is optimized by sampling edges from that list (probability
  proportional to weight) as positive pairs to attract, and uniformly random
  pairs as negative samples to repel, following a noise-contrastive objective
  with the same Cauchy low-dimensional kernel $1/(1 + \|y_i - y_j\|^2)$ used by
  t-SNE. This is the same negative-sampling trick used by LINE/node2vec-style
  graph embeddings, and it sidesteps the need to explicitly track which pairs
  are *not* edges.

  ## References

    * [Visualizing Large-scale and High-dimensional Data, Tang et al.](https://arxiv.org/abs/1602.00370)
  """
  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Neighbors.LargeVis, as: ANN

  opts = [
    num_components: [
      type: :pos_integer,
      default: 2,
      doc: "Dimension of the embedded space."
    ],
    num_neighbors: [
      type: :pos_integer,
      default: 15,
      doc: "The number of approximate nearest neighbors to use for each point."
    ],
    perplexity: [
      type: :pos_integer,
      default: 5,
      doc: """
      Target perplexity for the per-point Gaussian bandwidth search. Must be
      smaller than `:num_neighbors - 1`, since a point's effective neighbor
      count cannot exceed how many neighbors it actually has (the neighbor
      list includes the point itself, which is excluded from the affinities).
      """
    ],
    learning_rate: [
      type: {:or, [:pos_integer, :float]},
      default: 1.0,
      doc: "The learning rate for the embedding optimization."
    ],
    num_iters: [
      type: :pos_integer,
      default: 100,
      doc: "The number of optimization steps. Each step samples a batch of edges."
    ],
    batch_size: [
      type: :pos_integer,
      default: 500,
      doc: "The number of positive edges sampled at each optimization step."
    ],
    num_negative_samples: [
      type: :pos_integer,
      default: 5,
      doc: "The number of negative samples drawn for each positive edge."
    ],
    gamma: [
      type: {:or, [:pos_integer, :float]},
      default: 1.0,
      doc: "Weight of the negative (repulsive) samples relative to the positive ones."
    ],
    min_leaf_size: [
      type: :pos_integer,
      doc: """
      The minimum number of points in every leaf of the random projection forest.
      If not provided, it is set based on `:num_neighbors`.
      """
    ],
    num_trees: [
      type: :pos_integer,
      doc: """
      The number of trees in the random projection forest.
      If not provided, it is set based on the dataset size.
      """
    ],
    metric: [
      type: {:in, [:squared_euclidean, :euclidean]},
      default: :euclidean,
      doc: """
      The function that measures distance between two points during the
      k-NN graph construction. The affinities themselves are always computed
      on squared euclidean distances, as in t-SNE.
      """
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Used for random number generation in the k-NN graph construction and in
      the embedding optimization. If the key is not provided, it is set to
      `Nx.Random.key(System.system_time())`.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits LargeVis for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  Returns the embedded data, a tensor of shape `{num_samples, num_components}`.

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> x = Nx.concatenate([Nx.iota({20, 4}), Nx.add(Nx.iota({20, 4}), 100)])
      iex> y = Scholar.Manifold.LargeVis.fit(x, num_neighbors: 10, perplexity: 5, num_iters: 5, key: key)
      iex> Nx.shape(y)
      {40, 2}
  """
  deftransform fit(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected input tensor to have shape {num_samples, num_features}, " <>
              "got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)

    unless opts[:perplexity] < opts[:num_neighbors] - 1 do
      raise ArgumentError,
            "expected :perplexity to be smaller than :num_neighbors - 1 " <>
              "(the neighbor list includes the point itself, which is excluded), " <>
              "got perplexity: #{opts[:perplexity]} and num_neighbors: #{opts[:num_neighbors]}"
    end

    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_n(x, key, opts)
  end

  defnp fit_n(x, key, opts) do
    x = to_float(x)
    {num_samples, _} = Nx.shape(x)

    {graph, distances} = ANN.fit(x, ann_opts(opts, key))

    # The affinities follow the paper (and t-SNE): a Gaussian kernel on
    # squared distances. The metric only affects the neighbor search, whose
    # ranking is the same for :euclidean and :squared_euclidean.
    distances =
      case opts[:metric] do
        :euclidean -> distances * distances
        :squared_euclidean -> distances
      end

    # The k-NN graph lists each point as one of its own neighbors (distance 0).
    # Left in, that entry would dominate the softmax below, distorting the
    # perplexity search and producing self-edges whose gradient is zero, so
    # roughly half of the sampled positive pairs would be wasted. Masking its
    # distance to infinity zeroes its affinity and its sampling probability.
    self_mask = Nx.equal(graph, Nx.iota({num_samples, 1}))
    distances = Nx.select(self_mask, Nx.Constants.infinity(Nx.type(distances)), distances)

    sigmas = find_sigmas(distances, opts[:perplexity])
    p_cond = p_conditional(distances, sigmas)
    {rows, cols, weights} = build_edges(graph, p_cond)

    {y, key} =
      Nx.Random.uniform(key, -1.0, 1.0,
        shape: {num_samples, opts[:num_components]},
        type: Nx.type(x)
      )

    optimize(y, key, rows, cols, weights, num_samples, opts)
  end

  # :min_leaf_size and :num_trees fall back to Scholar.Neighbors.LargeVis's
  # own size-based defaults when not given, which only kicks in when the key
  # is absent from the options entirely (not merely nil).
  deftransformp ann_opts(opts, key) do
    [num_neighbors: opts[:num_neighbors], metric: opts[:metric], key: key]
    |> Keyword.merge(if opts[:min_leaf_size], do: [min_leaf_size: opts[:min_leaf_size]], else: [])
    |> Keyword.merge(if opts[:num_trees], do: [num_trees: opts[:num_trees]], else: [])
  end

  # Softmax of -distance / (2 * sigma^2) over the k neighbors. The self entry
  # the ANN graph carries has already had its distance set to infinity by
  # fit_n, so it contributes exp(-inf) = 0 here, the same effect t-SNE's
  # diagonal masking has on its dense conditional.
  defnp p_conditional(distances, sigmas) do
    arg = Nx.negate(distances) / (2 * Nx.reshape(sigmas, {:auto, 1}) ** 2)
    arg = arg - Nx.reduce_max(arg, axes: [1], keep_axes: true)
    p = Nx.exp(arg)
    p / Nx.sum(p, axes: [1], keep_axes: true)
  end

  defnp perplexity_from_p(p_row) do
    eps = Nx.Constants.epsilon(Nx.type(p_row))
    partials = Nx.select(p_row <= eps, 0, p_row * Nx.log2(p_row))
    2 ** Nx.negate(Nx.sum(partials, axes: [1]))
  end

  defnp binary_search_row(distances_row, target_perplexity, opts \\ []) do
    opts = keyword!(opts, tol: 1.0e-5, max_iters: 100, low: 1.0e-20, high: 1.0e4)
    row = Nx.new_axis(distances_row, 0)
    {low0, high0, max_iters, tol} = {opts[:low], opts[:high], opts[:max_iters], opts[:tol]}

    {low, high, _} =
      while {low = low0, high = high0,
             {max_iters, tol, perplexity_val = Nx.Constants.infinity(Nx.type(distances_row)), row,
              target_perplexity, i = 0}},
            i < max_iters and Nx.abs(perplexity_val - target_perplexity) > tol do
        mid = (low + high) / 2
        p_row = p_conditional(row, Nx.new_axis(mid, 0))
        perplexity_val = perplexity_from_p(p_row) |> Nx.reshape({})

        {low, high} =
          if perplexity_val > target_perplexity, do: {low, mid}, else: {mid, high}

        {low, high, {max_iters, tol, perplexity_val, row, target_perplexity, i + 1}}
      end

    (low + high) / 2
  end

  defnp find_sigmas(distances, target_perplexity) do
    n = Nx.axis_size(distances, 0)
    sigmas0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(distances)), {n})

    {sigmas, _} =
      while {sigmas = sigmas0, {distances, target_perplexity, i = 0}}, i < n do
        row = Nx.take(distances, i)
        sigma = binary_search_row(row, target_perplexity)
        sigmas = Nx.indexed_put(sigmas, Nx.reshape(i, {1, 1}), Nx.new_axis(sigma, -1))
        {sigmas, {distances, target_perplexity, i + 1}}
      end

    sigmas
  end

  # Forward edges (i, j, p_{j|i}) plus the same list with source/target
  # swapped. Summing both into a dense matrix is exactly P + P^T; the
  # optimization step below samples straight from this list and never needs
  # that dense matrix, or a canonical/deduplicated edge set, since duplicate
  # (i, j) entries just get sampled more often in proportion to their combined
  # weight, matching what deduplicating-and-summing them would give.
  defnp build_edges(graph, p_cond) do
    {n, k} = Nx.shape(graph)
    row = Nx.reshape(Nx.iota({n, k}, axis: 0), {n * k})
    col = Nx.reshape(graph, {n * k})
    w = Nx.reshape(p_cond, {n * k})

    rows = Nx.concatenate([row, col])
    cols = Nx.concatenate([col, row])
    weights = Nx.concatenate([w, w])
    {rows, cols, weights}
  end

  defnp optimize(y, key, rows, cols, weights, num_samples, opts) do
    num_iters = opts[:num_iters]
    learning_rate = opts[:learning_rate]
    edge_pool = Nx.iota({Nx.axis_size(rows, 0)})

    {y, _, _, _, _, _, _} =
      while {y, key, rows, cols, weights, edge_pool, i = 0}, i < num_iters do
        lr = Nx.max(learning_rate * (1.0 - i / num_iters), 0.05 * learning_rate)

        {y, key} =
          sgd_step(y, key, rows, cols, weights, edge_pool, num_samples, lr, opts)

        {y, key, rows, cols, weights, edge_pool, i + 1}
      end

    y
  end

  defnp sgd_step(y, key, rows, cols, weights, edge_pool, num_samples, lr, opts) do
    batch_size = opts[:batch_size]
    num_negative = opts[:num_negative_samples]
    gamma = opts[:gamma]
    eps = 1.0e-2
    clip = 5.0
    dim = Nx.axis_size(y, 1)

    {sampled, key} = Nx.Random.choice(key, edge_pool, weights, samples: batch_size)
    i_pos = Nx.take(rows, sampled)
    j_pos = Nx.take(cols, sampled)

    {j_neg, key} = Nx.Random.randint(key, 0, num_samples, shape: {batch_size, num_negative})

    yi = Nx.take(y, i_pos)
    yj = Nx.take(y, j_pos)
    diff_pos = yi - yj
    d2_pos = Nx.sum(diff_pos * diff_pos, axes: [1], keep_axes: true)
    attr_coef = 2.0 / (1.0 + d2_pos)
    delta_i_from_pos = attr_coef * Nx.negate(diff_pos)
    delta_j_from_pos = attr_coef * diff_pos

    yi_rep = Nx.new_axis(yi, 1) |> Nx.broadcast({batch_size, num_negative, dim})
    yneg = Nx.take(y, Nx.reshape(j_neg, {:auto})) |> Nx.reshape({batch_size, num_negative, dim})
    diff_neg = yi_rep - yneg
    d2_neg = Nx.sum(diff_neg * diff_neg, axes: [2], keep_axes: true)
    d2_neg = Nx.max(d2_neg, eps)
    rep_coef = gamma * 2.0 / (d2_neg * (1.0 + d2_neg))
    delta_i_from_neg = Nx.sum(rep_coef * diff_neg, axes: [1])

    updates_idx = Nx.concatenate([i_pos, j_pos, i_pos])

    updates_val =
      Nx.concatenate([delta_i_from_pos, delta_j_from_pos, delta_i_from_neg])
      |> Nx.clip(-clip, clip)
      |> Nx.multiply(lr)

    y = Nx.indexed_add(y, Nx.new_axis(updates_idx, -1), updates_val)
    {y, key}
  end
end
