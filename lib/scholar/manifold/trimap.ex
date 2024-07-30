defmodule Scholar.Manifold.Trimap do
  @moduledoc """
  TriMap: Large-scale Dimensionality Reduction Using Triplets.

  TriMap is a dimensionality reduction method that uses triplet constraints to form a low-dimensional embedding of a set of points.
  The triplet constraints are of the form "point i is closer to point j than point k".
  The triplets are sampled from the high-dimensional representation of the points and a weighting scheme is used to reflect the importance of each triplet.

  TriMap provides a significantly better global view of the data than the other dimensionality reduction methods such t-SNE, LargeVis.

  The global structure includes relative distances of the clusters, multiple scales in the data, and the existence of possible outliers.
  We define a global score to quantify the quality of an embedding in reflecting the global structure of the data.

  Based on: https://github.com/google-research/google-research/blob/master/trimap/trimap.py
  Source: https://arxiv.org/pdf/1910.00204.pdf
  """

  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Metrics.Distance

  @dim_pca 100
  @init_scale 0.01
  @init_momentum 0.5
  @final_momentum 0.8
  @switch_iter 250
  @min_gain 0.01
  @increase_gain 0.2
  @damp_gain 0.8

  opts_schema = [
    num_components: [
      type: :pos_integer,
      default: 2,
      doc: ~S"""
      Dimension of the embedded space.
      """
    ],
    num_inliers: [
      type: :pos_integer,
      default: 10,
      doc: ~S"""
      Number of inliers to sample.
      """
    ],
    num_outliers: [
      type: :pos_integer,
      default: 5,
      doc: ~S"""
      Number of outliers to sample.
      """
    ],
    num_random: [
      type: :pos_integer,
      default: 3,
      doc: ~S"""
      Number of random triplets to sample.
      """
    ],
    weight_temp: [
      type: :float,
      default: 0.5,
      doc: ~S"""
      Temperature for the tempered log.
      """
    ],
    learning_rate: [
      type: :float,
      default: 0.1,
      doc: ~S"""
      Learning rate.
      """
    ],
    num_iters: [
      type: :pos_integer,
      default: 400,
      doc: ~S"""
      Number of iterations.
      """
    ],
    init_embedding_type: [
      type: {:in, [0, 1]},
      default: 0,
      doc: ~S"""
      Method to initialize the embedding. 0 for PCA, 1 for random.
      """
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Random key used in Trimap.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ],
    triplets: [
      doc: ~S"""
      Triplets to use in the Trimap algorithm.
      """
    ],
    weights: [
      doc: ~S"""
      Weights to use in the Trimap algorithm.
      """
    ],
    init_embeddings: [
      doc: ~S"""
      Initial embeddings to use in the Trimap algorithm.
      """
    ],
    metric: [
      type: {:in, [:euclidean, :squared_euclidean, :cosine, :manhattan, :chebyshev]},
      default: :squared_euclidean,
      doc: ~S"""
      Metric used to compute the distances.
      """
    ],
    knn_algorithm: [
      type: {:in, [:auto, :nndescent, :large_vis, :brute]},
      default: :auto,
      doc: ~S"""
      Algorithm used to compute the nearest neighbors. Possible values:
      * `:nndescent` - Nearest Neighbors Descent. See `Scholar.Neighbors.NNDescent` for more details.

      * `:large_vis` - LargeVis algorithm. See `Scholar.Neighbors.LargeVis` for more details.

      * `:brute` - Brute force algorithm. See `Scholar.Neighbors.BruteKNN` for more details.

      * `:auto` - Automatically selects the algorithm based on the number of points.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  defnp tempered_log(x, t) do
    if Nx.abs(t - 1.0) < 1.0e-5 do
      Nx.log(x)
    else
      1.0 / (1.0 - t) * (x ** (1.0 - t) - 1.0)
    end
  end

  deftransformp handle_dist(x, y, opts) do
    apply(Distance, opts[:metric], [x, y, [axes: [-1]]])
  end

  defnp in1d(tensor1, tensor2) do
    order1 = Nx.argsort(tensor1)
    tensor1 = Nx.take(tensor1, order1)
    tensor2 = Nx.sort(tensor2)

    is_in = Nx.broadcast(Nx.u8(0), Nx.shape(tensor1))

    # binsearch which checks if the elements of tensor1 are in tensor2
    {is_in, _} =
      while {is_in, {tensor1, tensor2, prev = Nx.s64(-1), i = Nx.s64(0)}}, i < Nx.size(tensor1) do
        if i > 0 and prev == tensor1[i] do
          is_in = Nx.indexed_put(is_in, Nx.new_axis(i, 0), is_in[i - 1])
          {is_in, {tensor1, tensor2, prev, i + 1}}
        else
          {found?, _} =
            while {stop = Nx.u8(0),
                   {tensor1, tensor2, left = Nx.s64(0), right = Nx.size(tensor2) - 1, i}},
                  left <= right and not stop do
              mid = div(left + right, 2)

              if tensor1[i] == tensor2[mid] do
                {Nx.u8(1), {tensor1, tensor2, left, right, i}}
              else
                if tensor1[i] < tensor2[mid] do
                  {Nx.u8(0), {tensor1, tensor2, left, mid - 1, i}}
                else
                  {Nx.u8(0), {tensor1, tensor2, mid + 1, right, i}}
                end
              end
            end

          is_in = Nx.indexed_put(is_in, Nx.new_axis(i, 0), found?)
          prev = tensor1[i]
          {is_in, {tensor1, tensor2, prev, i + 1}}
        end
      end

    Nx.take(is_in, order1)
  end

  defnp rejection_sample(key, shape, rejects, opts \\ []) do
    final_samples = Nx.broadcast(Nx.s64(0), shape)

    {final_samples, key, _, _} =
      while {final_samples, key, rejects, i = Nx.s64(0)}, i < elem(shape, 0) do
        {samples, key} = Nx.Random.randint(key, 0, opts[:maxval], shape: {elem(shape, 1)})
        discard = in1d(samples, rejects[i])

        {samples, key, _, _, _} =
          while {samples, key, discard, rejects, i}, Nx.any(discard) do
            {new_samples, key} = Nx.Random.randint(key, 0, opts[:maxval], shape: {elem(shape, 1)})
            discard = in1d(new_samples, rejects[i]) or in1d(new_samples, samples)
            samples = Nx.select(discard, samples, new_samples)
            {samples, key, in1d(samples, rejects[i]), rejects, i}
          end

        final_samples = Nx.put_slice(final_samples, [i, 0], Nx.new_axis(samples, 0))
        {final_samples, key, rejects, i + 1}
      end

    {final_samples, key}
  end

  defnp sample_knn_triplets(key, neighbors, opts \\ []) do
    num_inliers = opts[:num_inliers]
    num_outliers = opts[:num_outliers]
    num_points = Nx.axis_size(neighbors, 0)

    num_inliers = min(num_inliers, Nx.axis_size(neighbors, 1) - 1)

    anchors =
      Nx.tile(Nx.iota({num_points, 1}), [1, num_inliers * num_outliers]) |> Nx.reshape({:auto, 1})

    inliers =
      Nx.tile(neighbors[[.., 1..num_inliers]], [1, num_outliers]) |> Nx.reshape({:auto, 1})

    {outliers, key} =
      rejection_sample(key, {num_points, num_inliers * num_outliers}, neighbors,
        maxval: num_points
      )

    outliers = Nx.reshape(outliers, {:auto, 1})

    {Nx.concatenate([anchors, inliers, outliers], axis: 1), key}
  end

  defnp sample_random_triplets(key, inputs, sig, opts \\ []) do
    num_points = Nx.axis_size(inputs, 0)
    num_random = opts[:num_random]

    anchors =
      Nx.tile(Nx.iota({num_points, 1}), [1, num_random]) |> Nx.reshape({:auto, 1})

    {pairs, key} =
      rejection_sample(key, {num_points * num_random, 2}, anchors, maxval: num_points)

    triplets = Nx.concatenate([anchors, pairs], axis: 1)
    anc = triplets[[.., 0]]
    sim = triplets[[.., 1]]
    out = triplets[[.., 2]]

    p_sim = -(handle_dist(inputs[anc], inputs[sim], opts) ** 2) / (sig[anc] * sig[sim])

    p_out = -(handle_dist(inputs[anc], inputs[out], opts) ** 2) / (sig[anc] * sig[out])

    flip = p_sim < p_out
    weights = p_sim - p_out

    pairs =
      Nx.select(
        Nx.tile(Nx.reshape(flip, {:auto, 1}), [1, 2]),
        Nx.reverse(pairs, axes: [1]),
        pairs
      )

    triplets = Nx.concatenate([anchors, pairs], axis: 1)
    {triplets, weights, key}
  end

  defnp find_scaled_neighbors(inputs, neighbors, opts \\ []) do
    {num_points, num_neighbors} = Nx.shape(neighbors)
    anchors = Nx.tile(Nx.iota({num_points, 1}), [1, num_neighbors]) |> Nx.flatten()
    hits = Nx.flatten(neighbors)

    distances =
      (handle_dist(inputs[anchors], inputs[hits], opts) ** 2) |> Nx.reshape({num_points, :auto})

    sigmas = Nx.max(Nx.mean(Nx.sqrt(distances[[.., 3..5]]), axes: [1]), 1.0e-10)

    scaled_distances = distances / (Nx.reshape(sigmas, {:auto, 1}) * sigmas[neighbors])
    sort_indices = Nx.argsort(scaled_distances, axis: 1)
    scaled_distances = Nx.take_along_axis(scaled_distances, sort_indices, axis: 1)
    sorted_neighbors = Nx.take_along_axis(neighbors, sort_indices, axis: 1)

    {scaled_distances, sorted_neighbors, sigmas}
  end

  defnp find_triplet_weights(inputs, triplets, neighbors, sigmas, distances, opts \\ []) do
    {num_points, num_inliers} = Nx.shape(neighbors)

    p_sim = -Nx.flatten(distances)

    num_outliers = div(Nx.axis_size(triplets, 0), num_points * num_inliers)

    p_sim =
      Nx.tile(Nx.reshape(p_sim, {num_points, num_inliers}), [1, num_outliers]) |> Nx.flatten()

    out_distances = handle_dist(inputs[triplets[[.., 0]]], inputs[triplets[[.., 2]]], opts) ** 2

    p_out = -out_distances / (sigmas[triplets[[.., 0]]] * sigmas[triplets[[.., 2]]])
    p_sim - p_out
  end

  defnp generate_triplets(key, inputs, opts \\ []) do
    num_inliers = opts[:num_inliers]
    num_random = opts[:num_random]
    weight_temp = opts[:weight_temp]
    num_points = Nx.axis_size(inputs, 0)

    num_extra = min(num_inliers + 50, num_points)

    neighbors =
      case opts[:knn_algorithm] do
        :brute ->
          model =
            Scholar.Neighbors.BruteKNN.fit(inputs,
              num_neighbors: num_extra,
              metric: opts[:metric]
            )

          {neighbors, _distances} = Scholar.Neighbors.BruteKNN.predict(model, inputs)
          neighbors

        :nndescent ->
          nndescent =
            Scholar.Neighbors.NNDescent.fit(inputs,
              num_neighbors: num_extra,
              tree_init?: false,
              metric: opts[:metric],
              tol: 1.0e-5,
              key: key
            )

          nndescent.nearest_neighbors

        :large_vis ->
          {neighbors, _distances} =
            Scholar.Neighbors.LargeVis.fit(inputs,
              num_neighbors: num_extra,
              metric: opts[:metric],
              key: key
            )

          neighbors

        :auto ->
          if Nx.axis_size(inputs, 0) <= 500 do
            model =
              Scholar.Neighbors.BruteKNN.fit(inputs,
                num_neighbors: num_extra,
                metric: opts[:metric]
              )

            {neighbors, _distances} = Scholar.Neighbors.BruteKNN.predict(model, inputs)
            neighbors
          else
            {neighbors, _distances} =
              Scholar.Neighbors.LargeVis.fit(inputs,
                num_neighbors: num_extra,
                metric: opts[:metric],
                key: key
              )

            neighbors
          end
      end

    neighbors = Nx.concatenate([Nx.iota({num_points, 1}), neighbors], axis: 1)

    {knn_distances, neighbors, sigmas} = find_scaled_neighbors(inputs, neighbors, opts)

    neighbors = neighbors[[.., 0..num_inliers]]
    knn_distances = knn_distances[[.., 0..num_inliers]]

    {triplets, key} =
      sample_knn_triplets(key, neighbors,
        num_outliers: opts[:num_outliers],
        num_inliers: num_inliers,
        num_points: num_points
      )

    weights =
      find_triplet_weights(
        inputs,
        triplets,
        neighbors[[.., 1..num_inliers]],
        sigmas,
        knn_distances[[.., 1..num_inliers]],
        opts
      )

    flip = weights < 0
    anchors = triplets[[.., 0]] |> Nx.reshape({:auto, 1})
    pairs = triplets[[.., 1..-1//1]]

    pairs =
      Nx.select(
        Nx.tile(Nx.reshape(flip, {:auto, 1}), [1, 2]),
        Nx.reverse(pairs, axes: [1]),
        pairs
      )

    triplets = Nx.concatenate([anchors, pairs], axis: 1)

    {triplets, weights, key} =
      if num_random > 0 do
        {random_triples, random_weights, key} = sample_random_triplets(key, inputs, sigmas, opts)

        {Nx.concatenate([triplets, random_triples], axis: 0),
         Nx.concatenate([weights, 0.1 * random_weights]), key}
      else
        {triplets, weights, key}
      end

    weights = weights - Nx.reduce_min(weights)
    weights = tempered_log(weights + 1.0, weight_temp)
    {triplets, weights, key}
  end

  # Update the embedding using delta-bar-delta.
  defnp update_embedding_dbd(embedding, grad, vel, gain, lr, iter_num) do
    gamma =
      Nx.select(iter_num > @switch_iter, @final_momentum, @init_momentum)

    gain =
      Nx.select(
        Nx.sign(vel) != Nx.sign(grad),
        gain + @increase_gain,
        Nx.max(gain * @damp_gain, @min_gain)
      )

    vel = gamma * vel - lr * gain * grad
    embedding = embedding + vel
    {embedding, gain, vel}
  end

  defnp trimap_metrics(embedding, triplets, weights) do
    anchors_points = embedding[triplets[[.., 0]]]
    pos_points = embedding[triplets[[.., 1]]]
    neg_points = embedding[triplets[[.., 2]]]

    sim_dist =
      1.0 + Distance.squared_euclidean(anchors_points, pos_points, axes: [-1])

    out_dist =
      1.0 + Distance.squared_euclidean(anchors_points, neg_points, axes: [-1])

    num_violated = Nx.sum(sim_dist > out_dist)
    loss = Nx.mean(weights * 1.0 / (1.0 + out_dist / sim_dist))

    {loss, num_violated}
  end

  defn trimap_loss({embedding, triplets, weights}) do
    {loss, _} = trimap_metrics(embedding, triplets, weights)
    loss
  end

  @doc """
  Embeds the given inputs using the TriMap algorithm.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}


  ## Examples

      iex> {inputs, key} = Nx.Random.uniform(Nx.Random.key(42), shape: {30, 5})
      iex> Scholar.Manifold.Trimap.transform(inputs, num_components: 2, num_inliers: 3, num_outliers: 1, key: key, knn_algorithm: :nndescent)
  """
  deftransform transform(inputs, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    {triplets, opts} = Keyword.pop(opts, :triplets, {})
    {weights, opts} = Keyword.pop(opts, :weights, {})
    {init_embeddings, opts} = Keyword.pop(opts, :init_embeddings, {})
    opts = Keyword.put(opts, :num_inliers, min(opts[:num_inliers], Nx.axis_size(inputs, 0) - 2))

    if opts[:num_inliers] <= 0 do
      raise ArgumentError, "Number of points must be greater than 2"
    end

    unless (Nx.rank(triplets) == Nx.rank(weights) and Nx.rank(triplets) == 0) or
             (Nx.rank(triplets) == 2 and Nx.rank(weights) == 1 and
                Nx.axis_size(triplets, 0) == Nx.axis_size(weights, 0)) do
      raise ArgumentError, "Triplets and weights must be either not initialized or have the same
      size of axis zero and rank of triplets must be 2 and rank of weights must be 1"
    end

    transform_n(
      inputs,
      key,
      triplets,
      weights,
      init_embeddings,
      opts
    )
  end

  defnp transform_n(inputs, key, triplets, weights, init_embeddings, opts \\ []) do
    {num_points, num_components} = Nx.shape(inputs)

    {triplets, weights, key, applied_pca?} =
      case triplets do
        {} ->
          {inputs, applied_pca} =
            if num_components > @dim_pca do
              inputs = inputs - Nx.mean(inputs, axes: [0])
              {u, s, vt} = Nx.LinAlg.SVD.svd(inputs, full_matrices: false)
              inputs = Nx.dot(u[[.., 0..@dim_pca]] * s[0..@dim_pca], vt[[0..@dim_pca, ..]])
              {inputs, Nx.u8(1)}
            else
              {inputs, Nx.u8(0)}
            end

          {triplets, weights, key} = generate_triplets(key, inputs, opts)
          {triplets, weights, key, applied_pca}

        _ ->
          {triplets, weights, key, Nx.u8(0)}
      end

    embeddings =
      case init_embeddings do
        {} ->
          cond do
            opts[:init_embedding_type] == 0 ->
              if applied_pca? do
                @init_scale * inputs[[.., 0..(opts[:num_components] - 1)]]
              else
                @init_scale *
                  Scholar.Decomposition.PCA.fit_transform(inputs,
                    num_components: opts[:num_components]
                  )
              end

            opts[:init_embedding_type] == 1 ->
              {random_embedding, _key} =
                Nx.Random.normal(key,
                  shape: {num_points, opts[:num_components]},
                  type: to_float_type(inputs)
                )

              random_embedding * @init_scale
          end

        _ ->
          init_embeddings
      end

    num_triplets = Nx.axis_size(triplets, 0)
    lr = opts[:learning_rate] * (num_points / num_triplets)

    vel = Nx.broadcast(Nx.tensor(0.0, type: to_float_type(embeddings)), Nx.shape(embeddings))
    gain = Nx.broadcast(Nx.tensor(1.0, type: to_float_type(embeddings)), Nx.shape(embeddings))

    {embeddings, _} =
      while {embeddings, {vel, gain, lr, triplets, weights, i = Nx.s64(0)}},
            i < opts[:num_iters] do
        gamma = if i < @switch_iter, do: @init_momentum, else: @final_momentum

        gradient =
          grad(embeddings + gamma * vel, fn x -> trimap_loss({x, triplets, weights}) end)

        {embeddings, vel, gain} = update_embedding_dbd(embeddings, gradient, vel, gain, lr, i)

        {embeddings, {vel, gain, lr, triplets, weights, i + 1}}
      end

    embeddings
  end
end
