defmodule Scholar.Manifold.LargeVisTest do
  use Scholar.Case, async: true
  alias Scholar.Manifold.LargeVis

  doctest LargeVis

  defp key, do: Nx.Random.key(42)

  # 4 well-separated Gaussian blobs in 10D. No scikit-learn equivalent exists
  # for LargeVis, so correctness here means the embedding actually preserves
  # cluster structure, checked via a neighborhood-purity metric: among each
  # point's nearest neighbors in the fitted 2D embedding, what fraction share
  # its original cluster label. A method producing a meaningless embedding
  # would score near 1 / num_clusters (random); a good one, near 1.0.
  defp blobs(num_per_cluster, num_clusters, num_features, spread, seed_offset) do
    centers =
      for c <- 0..(num_clusters - 1) do
        List.duplicate(0.0, num_features)
        |> List.replace_at(rem(c, num_features), spread * (c + 1))
      end

    {parts, labels} =
      Enum.reduce(0..(num_clusters - 1), {[], []}, fn c, {xs, ls} ->
        {noise, _} =
          Nx.Random.normal(Nx.Random.key(seed_offset + c), 0.0, 1.0,
            shape: {num_per_cluster, num_features},
            type: :f64
          )

        center = Nx.tensor(Enum.at(centers, c), type: :f64)
        {[Nx.add(noise, center) | xs], [List.duplicate(c, num_per_cluster) | ls]}
      end)

    x = Nx.concatenate(Enum.reverse(parts), axis: 0)
    labels = List.flatten(Enum.reverse(labels)) |> Nx.tensor()
    {x, labels}
  end

  defp neighborhood_purity(y, labels, k) do
    n = Nx.axis_size(y, 0)
    dist2 = Scholar.Metrics.Distance.pairwise_squared_euclidean(y)
    dist2_noself = Nx.put_diagonal(dist2, Nx.broadcast(Nx.Constants.infinity(:f64), {n}))
    {_, top_idx} = Nx.top_k(Nx.negate(dist2_noself), k: k)
    neighbor_labels = Nx.take(labels, top_idx)
    own_label = Nx.new_axis(labels, 1) |> Nx.broadcast({n, k})
    Nx.mean(Nx.equal(neighbor_labels, own_label)) |> Nx.to_number()
  end

  describe "fit" do
    test "produces the requested embedding shape" do
      x = Nx.iota({40, 5}) |> Nx.as_type(:f64)

      y =
        LargeVis.fit(x, num_neighbors: 10, perplexity: 5, num_iters: 5, key: key())

      assert Nx.shape(y) == {40, 2}
      assert Nx.type(y) == {:f, 64}
    end

    test "supports num_components other than 2" do
      x = Nx.iota({35, 4}) |> Nx.as_type(:f64)

      y =
        LargeVis.fit(x,
          num_neighbors: 8,
          perplexity: 4,
          num_iters: 5,
          num_components: 3,
          key: key()
        )

      assert Nx.shape(y) == {35, 3}
    end

    test "produces a finite embedding (no NaN/Inf)" do
      x = Nx.iota({40, 5}) |> Nx.as_type(:f64)

      y =
        LargeVis.fit(x, num_neighbors: 10, perplexity: 5, num_iters: 20, key: key())

      refute Nx.any(Nx.is_nan(y)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(y)) |> Nx.to_number() == 1
    end

    test "works inside jit" do
      x = Nx.iota({35, 4}) |> Nx.as_type(:f64)
      opts = [num_neighbors: 8, perplexity: 4, num_iters: 5, key: key()]

      direct = LargeVis.fit(x, opts)
      jitted = Nx.Defn.jit_apply(fn x -> LargeVis.fit(x, opts) end, [x])

      assert Nx.shape(jitted) == Nx.shape(direct)
      refute Nx.any(Nx.is_nan(jitted)) |> Nx.to_number() == 1
    end

    test "preserves f32 and casts integer input to f32" do
      opts = [num_neighbors: 8, perplexity: 4, num_iters: 5, key: key()]

      y_f32 = LargeVis.fit(Nx.iota({40, 4}) |> Nx.as_type(:f32), opts)
      assert Nx.type(y_f32) == {:f, 32}

      y_int = LargeVis.fit(Nx.iota({40, 4}), opts)
      assert Nx.type(y_int) == {:f, 32}
    end

    test "is deterministic for a fixed key" do
      x = Nx.iota({40, 4}) |> Nx.as_type(:f64)
      opts = [num_neighbors: 8, perplexity: 4, num_iters: 10, key: key()]

      assert_all_close(LargeVis.fit(x, opts), LargeVis.fit(x, opts), atol: 0.0, rtol: 0.0)
    end

    # Affinities are always computed on squared distances, and squaring is
    # monotone, so both metrics find the same neighbors and must produce the
    # same embedding up to floating-point roundtrip noise.
    test "euclidean and squared_euclidean metrics agree" do
      x = Nx.iota({40, 4}) |> Nx.as_type(:f64)
      opts = [num_neighbors: 8, perplexity: 4, num_iters: 5, key: key()]

      y_euclidean = LargeVis.fit(x, opts ++ [metric: :euclidean])
      y_squared = LargeVis.fit(x, opts ++ [metric: :squared_euclidean])

      assert Nx.shape(y_squared) == {40, 2}
      assert_all_close(y_euclidean, y_squared)
    end

    test "handles degenerate data with identical points" do
      x = Nx.broadcast(1.0, {40, 4}) |> Nx.as_type(:f64)

      y =
        LargeVis.fit(x, num_neighbors: 8, perplexity: 4, num_iters: 5, key: key())

      refute Nx.any(Nx.is_nan(y)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(y)) |> Nx.to_number() == 1
    end
  end

  describe "embedding quality" do
    test "separates well-clustered data far better than chance" do
      {x, labels} = blobs(25, 4, 10, 2.5, 700)

      y =
        LargeVis.fit(x,
          num_neighbors: 8,
          perplexity: 6,
          num_iters: 400,
          batch_size: 128,
          num_negative_samples: 5,
          key: Nx.Random.key(55)
        )

      purity = neighborhood_purity(y, labels, 5)

      # 1 / num_clusters would be the score of a meaningless embedding.
      assert purity > 0.9
    end
  end

  describe "errors" do
    test "raises for a non-rank-2 input" do
      assert_raise ArgumentError,
                   "expected input tensor to have shape {num_samples, num_features}, " <>
                     "got tensor with shape: {3}",
                   fn ->
                     LargeVis.fit(Nx.tensor([1, 2, 3]))
                   end
    end

    test "raises when perplexity is not smaller than num_neighbors - 1" do
      x = Nx.iota({20, 4}) |> Nx.as_type(:f64)

      message =
        "expected :perplexity to be smaller than :num_neighbors - 1 " <>
          "(the neighbor list includes the point itself, which is excluded), " <>
          "got perplexity: 4 and num_neighbors: 5"

      assert_raise ArgumentError, message, fn ->
        LargeVis.fit(x, num_neighbors: 5, perplexity: 4)
      end
    end
  end
end
