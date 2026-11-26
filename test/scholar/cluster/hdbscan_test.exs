defmodule Scholar.Cluster.HDBSCANTest do
  use Scholar.Case, async: true
  alias Scholar.Cluster.HDBSCAN
  doctest HDBSCAN

  # Three well separated groups of six. Chosen so that the merge heights in the single
  # linkage tree over the mutual reachability are distinct, which makes the dendrogram
  # unique and therefore forces any correct implementation to the same answer. That is
  # what lets the assertions below compare labels exactly rather than up to renumbering.
  # This holds at `min_samples: 3`; see the `min_samples: 4` test for the exception.
  defp blobs do
    Nx.tensor([
      [1.15, 1.585],
      [1.834, 0.694],
      [1.915, 1.287],
      [1.525, -0.246],
      [1.64, 0.51],
      [0.822, 0.832],
      [8.806, 2.281],
      [9.793, 1.606],
      [9.14, 2.101],
      [8.238, 1.596],
      [8.866, 1.012],
      [9.682, 2.307],
      [5.328, 7.954],
      [5.26, 8.128],
      [5.631, 9.082],
      [5.008, 8.702],
      [4.941, 8.269],
      [5.978, 9.37]
    ])
  end

  # The points of each cluster, ignoring which id the cluster was given.
  defp groups(labels) do
    labels
    |> Nx.to_flat_list()
    |> Enum.with_index()
    |> Enum.reject(fn {label, _} -> label == -1 end)
    |> Enum.group_by(fn {label, _} -> label end, fn {_, index} -> index end)
    |> Map.values()
    |> Enum.map(&Enum.sort/1)
    |> Enum.sort()
  end

  describe "fit" do
    # Reference: sklearn.cluster.HDBSCAN(min_cluster_size=3, min_samples=3).fit(x).labels_
    test "recovers three separated blobs" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3)

      assert model.labels ==
               Nx.tensor([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], type: :s32)
    end

    # Reference: sklearn.cluster.HDBSCAN(min_cluster_size=4, min_samples=4).fit(x).labels_
    # groups the same three blobs. At this `min_samples` two merges share a height, so
    # unlike the case above the tree is not unique and which blob gets id 1 and which
    # gets 2 is not determined. Compare the grouping rather than the ids.
    test "a larger min_cluster_size keeps the same three blobs" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 4, min_samples: 4)

      assert groups(model.labels) == [
               [0, 1, 2, 3, 4, 5],
               [6, 7, 8, 9, 10, 11],
               [12, 13, 14, 15, 16, 17]
             ]
    end

    test "a min_cluster_size larger than any blob leaves only noise" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 7, min_samples: 3)

      assert model.labels == Nx.broadcast(Nx.tensor(-1, type: :s32), {18})
    end

    test "labels are contiguous from zero, with -1 reserved for noise" do
      for min_cluster_size <- 2..7 do
        labels =
          blobs()
          |> HDBSCAN.fit(min_cluster_size: min_cluster_size, min_samples: 3)
          |> Map.fetch!(:labels)
          |> Nx.to_flat_list()

        clusters = labels |> Enum.reject(&(&1 == -1)) |> Enum.uniq() |> Enum.sort()

        assert Enum.all?(labels, &(&1 >= -1))
        assert clusters == Enum.to_list(0..(length(clusters) - 1)//1)
      end
    end

    test "min_samples defaults to min_cluster_size" do
      assert HDBSCAN.fit(blobs(), min_cluster_size: 4) ==
               HDBSCAN.fit(blobs(), min_cluster_size: 4, min_samples: 4)
    end

    test "min_samples changes the result" do
      loose = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 2)
      tight = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 12)

      refute loose.labels == tight.labels
    end
  end

  describe "metric" do
    # Each reference comes from sklearn.cluster.HDBSCAN(min_cluster_size=3, min_samples=3,
    # metric=...).fit(x).labels_ on the same points.
    test "euclidean is the default" do
      assert HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3) ==
               HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3, metric: :euclidean)
    end

    test "manhattan matches scikit-learn" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3, metric: {:minkowski, 1})

      assert model.labels ==
               Nx.tensor([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], type: :s32)
    end

    test "chebyshev matches scikit-learn" do
      model =
        HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3, metric: {:minkowski, :infinity})

      assert model.labels ==
               Nx.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], type: :s32)
    end

    # The metric feeds both the core distances and the mutual reachability. If it were only
    # threaded through one of them, or ignored, every metric would return the euclidean
    # answer. Chebyshev groups the same points but discovers them in a different order, and
    # cosine disagrees about the grouping itself.
    test "the metric is honored end to end" do
      euclidean = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3)

      chebyshev =
        HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3, metric: {:minkowski, :infinity})

      cosine = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3, metric: :cosine)

      refute euclidean.labels == chebyshev.labels
      refute euclidean.labels == cosine.labels
      assert_all_close(Nx.any(Nx.equal(cosine.labels, -1)), Nx.tensor(1))
    end

    test "accepts an anonymous function" do
      metric = fn a, b -> Scholar.Metrics.Distance.pairwise_squared_euclidean(a, b) end
      model = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3, metric: metric)

      # Squared euclidean is monotone in euclidean, so single linkage sees the same order
      # of merges and the clustering is unchanged.
      assert model.labels == HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3).labels
    end
  end

  # Every expected value here comes from sklearn.cluster.HDBSCAN on the same input.
  describe "degenerate input" do
    test "duplicate points merge at distance zero" do
      # A merge at distance 0 gives an infinite lambda. Getting there involves dividing by
      # the merge distance, which raises on some backends, so this is a regression test.
      x = Nx.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [9.0, 9.0], [9.0, 9.0], [9.0, 9.0]])
      model = HDBSCAN.fit(x, min_cluster_size: 2, min_samples: 2)

      assert model.labels == Nx.tensor([0, 0, 0, 1, 1, 1], type: :s32)
    end

    test "every point identical is all noise" do
      model = HDBSCAN.fit(Nx.broadcast(1.0, {8, 2}), min_cluster_size: 3)

      assert model.labels == Nx.broadcast(Nx.tensor(-1, type: :s32), {8})
    end

    test "evenly spaced collinear points are all noise" do
      x = Nx.tensor(Enum.map(0..9, &[&1 * 1.0, 0.0]))
      model = HDBSCAN.fit(x, min_cluster_size: 3)

      assert model.labels == Nx.broadcast(Nx.tensor(-1, type: :s32), {10})
    end

    test "a single feature" do
      x = Nx.tensor([[1.0], [1.2], [1.1], [8.0], [8.3], [8.1]])
      model = HDBSCAN.fit(x, min_cluster_size: 2, min_samples: 2)

      assert model.labels == Nx.tensor([0, 0, 0, 1, 1, 1], type: :s32)
    end

    test "the smallest accepted input" do
      x = Nx.tensor([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
      model = HDBSCAN.fit(x, min_cluster_size: 2)

      assert model.labels == Nx.tensor([-1, -1, -1], type: :s32)
    end

    test "more features than samples" do
      key = Nx.Random.key(1)
      {x, _} = Nx.Random.uniform(key, shape: {6, 30}, type: :f64)
      model = HDBSCAN.fit(x, min_cluster_size: 2, min_samples: 2)

      assert Nx.shape(model.labels) == {6}
    end

    test "integer input" do
      x = Nx.tensor([[1, 2], [2, 2], [2, 3], [1, 3], [8, 7], [8, 8], [7, 8], [7, 7]])
      model = HDBSCAN.fit(x, min_cluster_size: 3, min_samples: 2)

      assert model.labels == Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1], type: :s32)
    end

    # The distance metrics subtract and square in whatever dtype they are handed. Left in an
    # integer type that overflows, and on an unsigned type the subtraction wraps, which makes
    # the matrix asymmetric and then the clustering meaningless.
    test "every numeric dtype gives the same answer" do
      points = [[1, 2], [2, 2], [2, 3], [1, 3], [8, 7], [8, 8], [7, 8], [7, 7]]
      expected = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1], type: :s32)

      for type <- [:u8, :s8, :u16, :s16, :u32, :s32, :u64, :s64, :f16, :bf16, :f32, :f64] do
        model =
          Nx.tensor(points, type: type)
          |> HDBSCAN.fit(min_cluster_size: 3, min_samples: 2)

        assert model.labels == expected, "#{inspect(type)} disagrees"
      end
    end

    test "unsigned input works with every Minkowski exponent" do
      points = [
        [1, 2],
        [2, 2],
        [2, 3],
        [1, 3],
        [8, 7],
        [8, 8],
        [7, 8],
        [7, 7],
        [1, 8],
        [9, 1],
        [4, 4],
        [5, 5]
      ]

      signed =
        Nx.tensor(points, type: :s64)
        |> HDBSCAN.fit(min_cluster_size: 2, min_samples: 2, metric: {:minkowski, 1})

      for type <- [:u8, :u16, :u32, :u64], p <- [1, 3, :infinity] do
        model =
          Nx.tensor(points, type: type)
          |> HDBSCAN.fit(min_cluster_size: 2, min_samples: 2, metric: {:minkowski, p})

        assert Nx.shape(model.labels) == {12}

        if p == 1 do
          assert model.labels == signed.labels, "#{inspect(type)} disagrees with s64"
        end
      end
    end
  end

  describe "compilation and precision" do
    test "works with jit_apply" do
      jitted =
        Nx.Defn.jit_apply(&HDBSCAN.fit(&1, min_cluster_size: 3, min_samples: 3), [blobs()])

      direct = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3)

      assert jitted.labels == direct.labels
      assert_all_close(jitted.probabilities, direct.probabilities, atol: 0.0, rtol: 0.0)
    end

    test "is deterministic" do
      first = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3)

      for _ <- 1..3 do
        assert HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3).labels == first.labels
      end
    end

    test "f32 and f64 agree on well separated data" do
      x = blobs()

      assert HDBSCAN.fit(Nx.as_type(x, :f32), min_cluster_size: 3, min_samples: 3).labels ==
               HDBSCAN.fit(Nx.as_type(x, :f64), min_cluster_size: 3, min_samples: 3).labels
    end
  end

  describe "probabilities" do
    # Reference: sklearn.cluster.HDBSCAN(min_cluster_size=3, min_samples=3).fit(x).probabilities_
    test "match scikit-learn on three separated blobs" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3)

      assert_all_close(
        model.probabilities,
        Nx.tensor([
          0.9995818827546994,
          1.0,
          1.0,
          0.8297169836024464,
          1.0,
          0.933906690303436,
          0.9349886790485804,
          1.0,
          1.0,
          0.9208321594067206,
          0.7442533438899909,
          1.0,
          1.0,
          1.0,
          0.6837901010915983,
          0.795992941929263,
          1.0,
          0.42367886310518843
        ])
      )
    end

    # Reference: sklearn.cluster.HDBSCAN(min_cluster_size=2, min_samples=2).fit(x).probabilities_
    # The fifth point leaves its cluster halfway through the cluster's lifetime.
    test "a point that leaves early scores the fraction of its cluster's lifetime it lasted" do
      x = Nx.tensor([[1.0], [1.2], [1.1], [8.0], [8.3], [8.1]])
      model = HDBSCAN.fit(x, min_cluster_size: 2, min_samples: 2)

      assert model.labels == Nx.tensor([0, 0, 0, 1, 1, 1], type: :s32)
      assert_all_close(model.probabilities, Nx.tensor([1.0, 1.0, 1.0, 1.0, 0.5, 1.0]))
    end

    # Reference: sklearn.cluster.HDBSCAN(min_cluster_size=2, min_samples=2).fit(x).probabilities_
    # At this `min_cluster_size` the blobs split into sub-clusters that lose to their parent,
    # so the points inside them outlive the cluster that ends up owning them.
    test "a point that outlives the cluster owning it is a full member" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 2, min_samples: 2)

      assert model.labels ==
               Nx.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 3], type: :s32)

      assert_all_close(
        model.probabilities,
        Nx.tensor([
          0.3256787572416946,
          1.0,
          0.44674517481596904,
          0.3496548231486193,
          1.0,
          0.32554258533686375,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          0.4263703362431833,
          0.5356374544530236,
          1.0
        ])
      )
    end

    # Reference: sklearn.cluster.HDBSCAN(min_cluster_size=3, min_samples=2).fit(x).probabilities_
    # The seventh point drops out of the larger cluster at lambda 1 / 2.0, and that cluster
    # then ends by splitting in two at lambda 1 / 1.3, so the point lasted 0.65 of its
    # cluster's lifetime. A cluster that dies by splitting rather than by losing its last
    # point is the case that tells those two lambdas apart.
    test "a cluster's lifetime ends at the split that breaks it up" do
      x =
        Nx.tensor(
          [[0.0], [1.0], [2.05], [3.35], [4.4], [5.5], [7.5], [22.0], [23.05], [24.15]],
          type: :f64
        )

      model = HDBSCAN.fit(x, min_cluster_size: 3, min_samples: 2)

      assert model.labels == Nx.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], type: :s32)

      assert_all_close(
        model.probabilities,
        Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.65, 1.0, 1.0, 1.0])
      )
    end

    test "noise scores zero" do
      x = Nx.tensor(Enum.map(0..9, &[&1 * 1.0, 0.0]))
      model = HDBSCAN.fit(x, min_cluster_size: 3)

      assert model.labels == Nx.broadcast(Nx.tensor(-1, type: :s32), {10})
      assert_all_close(model.probabilities, Nx.broadcast(0.0, {10}))
    end

    # Duplicate points merge at distance 0, so they leave at an infinite lambda. That is a
    # full membership rather than a division of infinities.
    test "duplicate points are full members" do
      x = Nx.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [9.0, 9.0], [9.0, 9.0], [9.0, 9.0]])
      model = HDBSCAN.fit(x, min_cluster_size: 2, min_samples: 2)

      assert_all_close(model.probabilities, Nx.broadcast(1.0, {6}))
    end

    # A cluster's lifetime ends when its last member leaves, so whoever leaves last scores 1.
    test "every cluster keeps at least one point at full membership" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3)

      for cluster <- 0..2 do
        strongest =
          Nx.select(Nx.equal(model.labels, cluster), model.probabilities, 0.0)
          |> Nx.reduce_max()

        assert_all_close(strongest, Nx.tensor(1.0))
      end
    end

    test "probabilities stay between zero and one" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 2)

      in_range =
        Nx.logical_and(
          Nx.greater_equal(model.probabilities, 0.0),
          Nx.less_equal(model.probabilities, 1.0)
        )
        |> Nx.all()

      assert in_range == Nx.u8(1)
    end

    test "the type follows the input" do
      assert Nx.type(HDBSCAN.fit(Nx.as_type(blobs(), :f64), min_cluster_size: 3).probabilities) ==
               {:f, 64}

      assert Nx.type(HDBSCAN.fit(Nx.as_type(blobs(), :f32), min_cluster_size: 3).probabilities) ==
               {:f, 32}

      x = Nx.tensor([[1, 2], [2, 2], [2, 3], [1, 3], [8, 7], [8, 8], [7, 8], [7, 7]])

      assert Nx.type(HDBSCAN.fit(x, min_cluster_size: 3, min_samples: 2).probabilities) ==
               {:f, 32}
    end
  end

  describe "errors" do
    test "x must be rank 2" do
      assert_raise ArgumentError,
                   "expected x to have shape {num_samples, num_features}, got tensor with shape: {3}",
                   fn -> HDBSCAN.fit(Nx.tensor([1, 2, 3])) end
    end

    test "min_cluster_size must be at least 2" do
      assert_raise ArgumentError, "expected :min_cluster_size to be at least 2, got: 1", fn ->
        HDBSCAN.fit(blobs(), min_cluster_size: 1)
      end
    end

    test "x must have at least 3 samples" do
      assert_raise ArgumentError, "expected x to have at least 3 samples, got: 2", fn ->
        HDBSCAN.fit(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), min_cluster_size: 2)
      end
    end

    test "min_samples may not exceed the number of samples" do
      assert_raise ArgumentError,
                   "expected :min_samples to be at most the number of samples (18), got: 19",
                   fn -> HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 19) end
    end
  end
end
