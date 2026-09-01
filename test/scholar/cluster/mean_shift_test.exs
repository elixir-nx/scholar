defmodule Scholar.Cluster.MeanShiftTest do
  use Scholar.Case, async: true
  alias Scholar.Cluster.MeanShift
  doctest MeanShift

  defp blobs do
    Nx.tensor([
      [1.0, 1.0],
      [1.2, 1.1],
      [0.9, 1.05],
      [8.0, 8.0],
      [8.1, 8.2],
      [7.9, 7.8],
      [1.0, 8.0],
      [1.1, 8.1]
    ])
  end

  describe "fit" do
    test "fit - all defaults" do
      # Expected values from scikit-learn 1.6.1 on this data.
      model = MeanShift.fit(blobs(), bandwidth: 1.0) |> MeanShift.prune()

      assert model.num_clusters == Nx.u32(3)
      assert model.labels == Nx.s32([1, 1, 1, 0, 0, 0, 2, 2])

      assert_all_close(
        model.cluster_centers,
        Nx.tensor([[8.0, 8.0], [1.0333334, 1.05], [1.05, 8.05]])
      )
    end

    test "fit with a bandwidth wide enough to hold every sample" do
      # Expected values from scikit-learn 1.6.1 on this data.
      model = MeanShift.fit(blobs(), bandwidth: 20.0) |> MeanShift.prune()

      assert model.num_clusters == Nx.u32(1)
      assert model.labels == Nx.s32([0, 0, 0, 0, 0, 0, 0, 0])
      assert_all_close(model.cluster_centers, Nx.tensor([[3.65, 5.40625]]))
    end

    test "fit with a bandwidth narrower than the closest pair" do
      model = MeanShift.fit(blobs(), bandwidth: 0.05) |> MeanShift.prune()

      # every sample keeps to itself, so the centers are the samples reordered
      assert model.num_clusters == Nx.u32(8)
      assert model.labels == Nx.s32([6, 3, 7, 1, 0, 2, 5, 4])
    end

    test "fit with cluster_all disabled" do
      # Expected values from scikit-learn 1.6.1 on this data.
      x =
        Nx.tensor([
          [2.2, -2.9],
          [5.4, -4.8],
          [1.0, -1.7],
          [3.6, -2.9],
          [1.2, -0.4],
          [5.1, 1.9]
        ])

      clustered = MeanShift.fit(x, bandwidth: 2.0, cluster_all: true) |> MeanShift.prune()
      assert clustered.labels == Nx.s32([0, 1, 0, 0, 0, 2])

      # the fifth sample drifted into the first mode but stayed further than the
      # bandwidth from where that mode settled
      loose = MeanShift.fit(x, bandwidth: 2.0, cluster_all: false) |> MeanShift.prune()
      assert loose.labels == Nx.s32([0, 1, 0, 0, -1, 2])

      assert_all_close(
        loose.cluster_centers,
        Nx.tensor([[2.2666667, -2.5], [5.4, -4.8], [5.1, 1.9]])
      )
    end

    test "fit with samples that are all the same point" do
      # Expected values from scikit-learn 1.6.1 on this data.
      model =
        MeanShift.fit(Nx.tensor([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]]), bandwidth: 1.0)
        |> MeanShift.prune()

      assert model.num_clusters == Nx.u32(1)
      assert model.labels == Nx.s32([0, 0, 0])
      assert_all_close(model.cluster_centers, Nx.tensor([[3.0, 3.0]]))
    end

    test "fit with a single sample" do
      model = MeanShift.fit(Nx.tensor([[2.0, 5.0]]), bandwidth: 1.0) |> MeanShift.prune()

      assert model.num_clusters == Nx.u32(1)
      assert model.labels == Nx.s32([0])
      assert_all_close(model.cluster_centers, Nx.tensor([[2.0, 5.0]]))
    end

    test "fit with one feature" do
      # Expected values from scikit-learn 1.6.1 on this data.
      model =
        MeanShift.fit(Nx.tensor([[1.0], [1.1], [5.0], [5.2]]), bandwidth: 0.5)
        |> MeanShift.prune()

      assert model.labels == Nx.s32([1, 1, 0, 0])
      assert_all_close(model.cluster_centers, Nx.tensor([[5.1], [1.05]]))
    end

    test "fit stops at max_iterations" do
      {x, _} = Nx.Random.uniform(Nx.Random.key(7), shape: {40, 2})
      x = Nx.multiply(x, 10)

      truncated = MeanShift.fit(x, bandwidth: 2.5, max_iterations: 1)
      settled = MeanShift.fit(x, bandwidth: 2.5)

      assert truncated.iterations == Nx.u32(1)
      assert settled.iterations == Nx.u32(8)

      # seeds cut off early have not merged yet, so more of them survive
      assert Nx.greater(truncated.num_clusters, settled.num_clusters) == Nx.u8(1)
    end

    test "fit from a given set of seeds" do
      x = Nx.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0]])

      model =
        MeanShift.fit(x, bandwidth: 1.0, seeds: Nx.tensor([[0.0, 0.0], [5.0, 5.0]]))
        |> MeanShift.prune()

      assert model.num_clusters == Nx.u32(2)
      assert model.labels == Nx.s32([0, 0, 0, 1, 1])
      assert_all_close(model.cluster_centers, Nx.tensor([[0.0333333, 0.0333333], [5.05, 5.0]]))
    end

    test "fit drops seeds that never reach a sample" do
      # Expected values from scikit-learn 1.6.1 on this data.
      x = Nx.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

      model =
        MeanShift.fit(x, bandwidth: 1.0, seeds: Nx.tensor([[0.0, 0.0], [99.0, 99.0]]))
        |> MeanShift.prune()

      assert model.num_clusters == Nx.u32(1)
      assert model.labels == Nx.s32([0, 0, 0])
      assert_all_close(model.cluster_centers, Nx.tensor([[0.0333333, 0.0333333]]))
    end

    test "fit keeps the type of the input" do
      f64 = MeanShift.fit(Nx.as_type(blobs(), :f64), bandwidth: 1.0)
      assert Nx.type(f64.cluster_centers) == {:f, 64}

      f32 = MeanShift.fit(blobs(), bandwidth: 1.0)
      assert Nx.type(f32.cluster_centers) == {:f, 32}
    end

    test "fit with samples and seeds of different types" do
      seeds = Nx.tensor([[1.0, 1.0], [8.0, 8.0]])

      # the seeds carry the loop's accumulator, so a wider sample type has to
      # widen them too rather than fail to match
      wide = MeanShift.fit(Nx.as_type(blobs(), :f64), bandwidth: 1.0, seeds: seeds)
      assert Nx.type(wide.cluster_centers) == {:f, 64}

      narrow =
        MeanShift.fit(blobs(), bandwidth: 1.0, seeds: Nx.as_type(seeds, :f64))

      assert Nx.type(narrow.cluster_centers) == {:f, 64}
      assert_all_close(wide.cluster_centers, narrow.cluster_centers)
    end

    test "works with jit_apply" do
      direct = MeanShift.fit(blobs(), bandwidth: 1.0)
      jitted = Nx.Defn.jit_apply(&MeanShift.fit/2, [blobs(), [bandwidth: 1.0]])

      assert jitted.labels == direct.labels
      assert jitted.num_clusters == direct.num_clusters
      assert_all_close(jitted.cluster_centers, direct.cluster_centers)
    end
  end

  test "prune" do
    model = MeanShift.fit(blobs(), bandwidth: 1.0)
    pruned = MeanShift.prune(model)

    assert model.num_clusters == Nx.u32(3)
    assert pruned.num_clusters == model.num_clusters

    # fit keeps one row per seed, so pruning has to leave exactly the rows that
    # did not lose to a stronger center
    assert_all_close(
      pruned.cluster_centers,
      Nx.tensor([[8.0, 8.0], [1.0333334, 1.05], [1.05, 8.05]])
    )

    # and the labels have to keep pointing at the same centers after renumbering
    assert_all_close(
      Nx.take(pruned.cluster_centers, pruned.labels),
      Nx.take(model.cluster_centers, model.labels)
    )
  end

  describe "errors" do
    test "x that is not a matrix" do
      assert_raise ArgumentError,
                   "expected x to have shape {num_samples, num_features}, got: {3}",
                   fn -> MeanShift.fit(Nx.tensor([1, 2, 3]), bandwidth: 1.0) end
    end

    test "seeds that do not match the number of features" do
      assert_raise ArgumentError,
                   "expected seeds to have shape {num_seeds, 2}, got: {1, 3}",
                   fn ->
                     MeanShift.fit(blobs(), bandwidth: 1.0, seeds: Nx.tensor([[1.0, 2.0, 3.0]]))
                   end
    end

    test "prune with nothing left to keep" do
      model =
        MeanShift.fit(Nx.tensor([[0.0, 0.0]]), bandwidth: 1.0, seeds: Nx.tensor([[99.0, 99.0]]))

      assert model.num_clusters == Nx.u32(0)
      assert model.labels == Nx.s32([-1])

      assert_raise ArgumentError,
                   "the model has no clusters to keep, every seed was further than the " <>
                     "bandwidth from all samples",
                   fn -> MeanShift.prune(model) end
    end
  end
end
