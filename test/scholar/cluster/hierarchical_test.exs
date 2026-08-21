defmodule Scholar.Cluster.HierarchicalTest do
  use Scholar.Case, async: true

  alias Scholar.Cluster.Hierarchical

  doctest Hierarchical

  describe "basic example" do
    test "works" do
      # This diagram represents data. `0` appears at the coordinates (1, 5). The 0th entry of data
      # is `[1, 5]`. Same for 1, etc.
      #
      #   5 | 0 1   3 4
      #   4 | 2       5
      #   3 |
      #   2 | 6
      #   1 | 7 8
      #   0 +-+-+-+-+-+
      #     0 1 2 3 4 5
      data = Nx.tensor([[1, 5], [2, 5], [1, 4], [4, 5], [5, 5], [5, 4], [1, 2], [1, 1], [2, 1]])

      # This diagram represents the sequence of expected merges. The data starts off with all
      # points as singleton clades. The algorithm builds a chain of nearest neighbors (a point,
      # its nearest neighbor, that neighbor's nearest, ...) and merges as soon as the last two
      # entries in the chain are each other's nearest neighbor, then keeps extending the same
      # chain from what is left rather than restarting. That is why, below, clade 9: [0, 1]
      # immediately pulls in point 2 (its nearest neighbor is now clade 9) before the algorithm
      # moves on to points 3 and 4, rather than growing every clade one round at a time.
      #
      #       0   1   2   3   4   5   6   7   8
      #    8: [0] [1] [2] [3] [4] [5] [6] [7] [8]
      #       9    2   3   4   5   6   7   8
      #    9: [01] [2] [3] [4] [5] [6] [7] [8]
      #       ----
      #       10   3   4   5   6   7   8
      #   10: [012] [3] [4] [5] [6] [7] [8]
      #       -----
      #       10    11   5   6   7   8
      #   11: [012] [34] [5] [6] [7] [8]
      #                ----
      #       10    12    6   7   8
      #   12: [012] [345] [6] [7] [8]
      #             -----
      #       10    12    13   8
      #   13: [012] [345] [67] [8]
      #                    ----
      #       10    12    14
      #   14: [012] [345] [678]
      #                   -----
      #       15       14
      #   15: [012345] [678]
      #       --------
      #       16
      #   16: [012345678]
      #       -----------
      model = Hierarchical.fit(data, dissimilarity: :euclidean, linkage: :single)

      # The dendrogram formation part of the algorithm should've formed the following clades,
      # dissimilarities, and sizes (which collectively form the dendrogram).
      assert model.clades ==
               Nx.tensor([
                 [0, 1],
                 [2, 9],
                 [3, 4],
                 [5, 11],
                 [6, 7],
                 [8, 13],
                 [10, 12],
                 [14, 15]
               ])

      assert model.dissimilarities ==
               Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])

      assert model.sizes == Nx.tensor([2, 3, 2, 3, 2, 3, 6, 9])

      # The clustering part of the algorithm uses the `cluster_by: [num_clusters: 3]` option to
      # take the model and form 3 clusters.
      labels_map = Hierarchical.labels_map(model, cluster_by: [num_clusters: 3])
      assert labels_map == %{0 => [0, 1, 2], 1 => [3, 4, 5], 2 => [6, 7, 8]}

      # We can also return a list of each datum's cluster label.
      labels_list = Hierarchical.labels_list(model, cluster_by: [num_clusters: 3])
      assert labels_list == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    end
  end

  describe "linkages" do
    setup do
      %{data: Nx.tensor([[1, 5], [2, 5], [1, 4], [4, 5], [5, 5], [5, 4], [1, 2], [1, 1], [2, 1]])}
    end

    test "average", %{data: data} do
      model = Hierarchical.fit(data, linkage: :average)

      assert model.dissimilarities ==
               Nx.tensor([
                 1.0,
                 1.0,
                 1.0,
                 1.2071068286895752,
                 1.2071068286895752,
                 1.2071068286895752,
                 3.396751642227173,
                 4.092065334320068
               ])
    end

    test "complete", %{data: data} do
      model = Hierarchical.fit(data, linkage: :complete)

      assert model.dissimilarities ==
               Nx.tensor([
                 1.0,
                 1.0,
                 1.0,
                 # sqrt(2)
                 1.4142135381698608,
                 1.4142135381698608,
                 1.4142135381698608,
                 # sqrt(17)
                 4.123105525970459,
                 # 4 * sqrt(2)
                 5.656854152679443
               ])
    end

    test "single", %{data: data} do
      model = Hierarchical.fit(data, linkage: :single)
      assert model.dissimilarities == Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    end

    test "ward", %{data: data} do
      model = Hierarchical.fit(data, linkage: :ward)

      # Reference values taken from SciPy (scipy.cluster.hierarchy.linkage)
      assert_all_close(
        model.dissimilarities,
        Nx.tensor([
          1.0,
          1.0,
          1.0,
          1.29099445,
          1.29099445,
          1.29099445,
          5.77350269,
          7.45355992
        ])
      )
    end

    test "weighted", %{data: data} do
      model = Hierarchical.fit(data, linkage: :weighted)

      assert model.dissimilarities ==
               Nx.tensor([
                 1.0,
                 1.0,
                 1.0,
                 1.2071068286895752,
                 1.2071068286895752,
                 1.2071068286895752,
                 3.32379412651062,
                 4.1218791007995605
               ])
    end

    test "ward does not take the square root of a negative number" do
      # Merged clades are masked out of the matrix rather than blanked, so their
      # rows keep stale values that ward's update still reads. The Lance-Williams
      # term can then go negative for a dead column, and the square root of that
      # raises on the binary backend and is a silent NaN under EXLA.
      data =
        Nx.tensor([
          [0.0, 0.0],
          [2.0, 1.0],
          [2.0, 0.0],
          [2.0, 1.0],
          [2.0, 1.0],
          [1.0, 1.0],
          [1.0, 0.0],
          [0.0, 2.0]
        ])

      model = Hierarchical.fit(data, linkage: :ward)

      refute model.dissimilarities |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 1
    end
  end

  describe "cluster labels" do
    setup do
      %{model: Hierarchical.fit(Nx.tensor([[2], [7], [9], [0], [3]]))}
    end

    test "cluster by height", %{model: model} do
      labels_map = Hierarchical.labels_map(model, cluster_by: [height: 2.5])
      assert labels_map == %{0 => [0, 3, 4], 1 => [1, 2]}
      labels_list = Hierarchical.labels_list(model, cluster_by: [height: 2.5])
      assert labels_list == [0, 1, 1, 0, 0]
    end

    test "cluster by number of clusters", %{model: model} do
      labels_map = Hierarchical.labels_map(model, cluster_by: [num_clusters: 3])
      assert labels_map == %{0 => [0, 3, 4], 1 => [1], 2 => [2]}
      labels_list = Hierarchical.labels_list(model, cluster_by: [num_clusters: 3])
      assert labels_list == [0, 1, 2, 0, 0]
    end

    test "remaps clade ids after sorting dendrogram rows" do
      {data, _key} = Nx.Random.uniform(Nx.Random.key(0), shape: {50, 2})
      model = Nx.Defn.jit_apply(&Hierarchical.fit/1, [data])

      assert model.clades
             |> Nx.to_list()
             |> Enum.with_index(model.num_points)
             |> Enum.all?(fn {children, clade_id} ->
               Enum.all?(children, &(&1 < clade_id))
             end)

      labels = Hierarchical.labels_list(model, cluster_by: [num_clusters: 5])
      assert length(labels) == 50
      assert labels |> Enum.uniq() |> length() == 5
    end

    test "tied merge heights still leave every dendrogram row in ascending order" do
      # Duplicate points force merges to tie. Remapping clade ids to their final,
      # sorted-by-dissimilarity numbering is a permutation, not a monotonic one, so
      # on a tie it can swap the two children of a row and leave it descending.
      data =
        Nx.tensor([
          [0.0, 0.0],
          [0.0, 1.0],
          [0.0, 0.0],
          [1.0, 1.0],
          [1.0, 0.0],
          [1.0, 1.0],
          [2.0, 0.0],
          [2.0, 1.0]
        ])

      model = Hierarchical.fit(data, linkage: :complete)

      assert model.dissimilarities |> Nx.to_flat_list() |> Enum.uniq() |> length() <
               model.num_points - 1,
             "expected tied merge heights, which is what this test is about"

      assert model.clades
             |> Nx.to_list()
             |> Enum.all?(fn [left, right] -> left < right end)
    end

    test "tied nearest neighbors still merge every point exactly once" do
      # Integer coordinates over a small range, so distances tie constantly and a
      # clade's nearest neighbor is very often not unique. The chain extension can
      # then walk back onto a clade it already holds, and the duplicate outlives the
      # merge that consumed it, so it gets merged a second time: a clade is used
      # twice, the sizes stop adding up, and the tree never closes over every point.
      data =
        Nx.tensor([
          [0.0, 2.0, 2.0],
          [1.0, 3.0, 1.0],
          [2.0, 3.0, 1.0],
          [0.0, 0.0, 0.0],
          [0.0, 1.0, 3.0],
          [3.0, 0.0, 3.0],
          [1.0, 1.0, 1.0],
          [1.0, 1.0, 3.0],
          [2.0, 2.0, 1.0],
          [1.0, 0.0, 1.0],
          [1.0, 1.0, 3.0],
          [1.0, 0.0, 0.0],
          [3.0, 2.0, 3.0],
          [0.0, 3.0, 1.0],
          [1.0, 1.0, 0.0],
          [3.0, 2.0, 3.0],
          [2.0, 0.0, 0.0],
          [0.0, 1.0, 0.0]
        ])

      model = Hierarchical.fit(data, linkage: :single)

      # The last merge has to gather every point.
      assert Nx.to_number(model.sizes[-1]) == model.num_points

      # And no clade may be merged into two different parents.
      children = model.clades |> Nx.to_flat_list()
      assert length(children) == children |> Enum.uniq() |> length()
    end
  end

  describe "precomputed dissimilarity" do
    setup do
      %{
        data: Nx.tensor([[1, 5], [2, 5], [1, 4], [4, 5], [5, 5], [5, 4], [1, 2], [1, 1], [2, 1]]),
        # A dissimilarity that does not come from coordinates and violates the triangle
        # inequality, since d(0, 2) = 9 is greater than d(0, 1) + d(1, 2) = 2. It cannot be
        # produced by a euclidean fit. Reference values from
        # `scipy.cluster.hierarchy.linkage(squareform(d), method=...)`.
        dissimilarities:
          Nx.tensor([
            [0.0, 1.0, 9.0, 8.0, 7.0],
            [1.0, 0.0, 1.0, 8.0, 7.0],
            [9.0, 1.0, 0.0, 8.0, 7.0],
            [8.0, 8.0, 8.0, 0.0, 2.0],
            [7.0, 7.0, 7.0, 2.0, 0.0]
          ])
      }
    end

    for linkage <- [:average, :complete, :single, :ward, :weighted] do
      test "#{linkage} matches computing the dissimilarities internally", %{data: data} do
        linkage = unquote(linkage)
        dissimilarities = Scholar.Metrics.Distance.pairwise_euclidean(data)

        assert Hierarchical.fit(dissimilarities, dissimilarity: :precomputed, linkage: linkage) ==
                 Hierarchical.fit(data, dissimilarity: :euclidean, linkage: linkage)
      end
    end

    test "single matches scipy", %{dissimilarities: d} do
      model = Hierarchical.fit(d, dissimilarity: :precomputed, linkage: :single)

      assert model.dissimilarities == Nx.tensor([1.0, 1.0, 2.0, 7.0])
      assert model.sizes == Nx.tensor([2, 3, 2, 5])
      assert model.num_points == 5
    end

    test "complete matches scipy", %{dissimilarities: d} do
      model = Hierarchical.fit(d, dissimilarity: :precomputed, linkage: :complete)

      assert model.dissimilarities == Nx.tensor([1.0, 2.0, 8.0, 9.0])
    end

    test "average matches scipy", %{dissimilarities: d} do
      model = Hierarchical.fit(d, dissimilarity: :precomputed, linkage: :average)

      assert model.dissimilarities == Nx.tensor([1.0, 2.0, 5.0, 7.5])
    end

    test "weighted matches scipy", %{dissimilarities: d} do
      model = Hierarchical.fit(d, dissimilarity: :precomputed, linkage: :weighted)

      assert model.dissimilarities == Nx.tensor([1.0, 2.0, 5.0, 7.5])
    end

    test "the diagonal is ignored", %{dissimilarities: d} do
      polluted = Nx.put_diagonal(d, Nx.tensor([100.0, 100.0, 100.0, 100.0, 100.0]))

      assert Hierarchical.fit(polluted, dissimilarity: :precomputed) ==
               Hierarchical.fit(d, dissimilarity: :precomputed)
    end

    test "labels can be derived from a precomputed model", %{dissimilarities: d} do
      model = Hierarchical.fit(d, dissimilarity: :precomputed, linkage: :single)

      assert Hierarchical.labels_list(model, cluster_by: [num_clusters: 2]) == [0, 0, 0, 1, 1]
    end

    test "works with jit_apply", %{data: data} do
      dissimilarities = Scholar.Metrics.Distance.pairwise_euclidean(data)

      jitted =
        Nx.Defn.jit_apply(
          fn d -> Hierarchical.fit(d, dissimilarity: :precomputed, linkage: :single) end,
          [dissimilarities]
        )

      assert jitted == Hierarchical.fit(dissimilarities, dissimilarity: :precomputed)
    end
  end

  describe "non-finite dissimilarities" do
    # Regression tests for a real hang: with a non-finite dissimilarity (from :nan or
    # :infinity), argmin's tie-breaking can point two clades at each other without either
    # being picked as the other's mutual nearest neighbor, so no merge ever happens. This is
    # impossible for finite dissimilarities, where the globally closest pair of clades is
    # always mutually nearest to each other, guaranteeing at least one merge every round.
    # When it happens the loop now stops instead of running forever, and the merges it could
    # not make are reported as NaN dissimilarities with clade -1 and size 0.
    test "an infinite dissimilarity between the last two clades reports an incomplete merge" do
      # Point 3 is infinitely far from everyone. Once points 0, 1 and 2 (mutually close)
      # have merged, only two clades remain: {0, 1, 2} and {3}, at distance infinity, and
      # there is no finite distance left to justify merging them over any other pairing.
      d =
        Nx.tensor([
          [0.0, 1.0, 1.0, :infinity],
          [1.0, 0.0, 1.4142135623730951, :infinity],
          [1.0, 1.4142135623730951, 0.0, :infinity],
          [:infinity, :infinity, :infinity, 0.0]
        ])

      model = Hierarchical.fit(d, dissimilarity: :precomputed, linkage: :single)

      assert model.num_points == 4
      # The two finite merges are made and kept, in ascending order.
      assert Nx.to_flat_list(model.dissimilarities) |> Enum.take(2) == [1.0, 1.0]
      # The merge that could not be made is reported instead of guessed.
      assert Nx.to_number(Nx.is_nan(model.dissimilarities[-1])) == 1
      assert Nx.to_flat_list(model.clades[-1]) == [-1, -1]
      assert Nx.to_number(model.sizes[-1]) == 0
    end

    test "a NaN coordinate still makes every merge" do
      # NaN alone does not stall the loop: argmin keeps picking a mutual pair, so every
      # merge is made. The dissimilarities really are NaN here, since the distances are,
      # which is why an incomplete merge is identified by its clade and size, not by NaN.
      x = Nx.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [5.0, 5.0], [5.1, 5.0], [:nan, 0.0]])

      model = Hierarchical.fit(x)

      assert model.num_points == 6
      assert Nx.to_number(Nx.all(Nx.not_equal(model.clades, -1))) == 1
      assert Nx.to_number(Nx.all(Nx.greater(model.sizes, 0))) == 1
    end

    test "several unreachable clades report every merge that could not be made" do
      # Only 0 and 1 have a finite distance to anything. Points 2 and 3 are infinitely
      # far from everyone including each other, so two of the three merges have nothing
      # to justify them, not just the last one.
      d =
        Nx.tensor([
          [0.0, 1.0, :infinity, :infinity],
          [1.0, 0.0, :infinity, :infinity],
          [:infinity, :infinity, 0.0, :infinity],
          [:infinity, :infinity, :infinity, 0.0]
        ])

      model = Hierarchical.fit(d, dissimilarity: :precomputed, linkage: :single)

      assert Nx.to_flat_list(model.sizes) == [2, 0, 0]
      assert Nx.to_flat_list(model.dissimilarities) |> Enum.take(1) == [1.0]
      assert Nx.to_number(Nx.all(Nx.is_nan(model.dissimilarities[1..2]))) == 1
      assert Nx.to_flat_list(model.clades[1..2]) == [-1, -1, -1, -1]
    end
  end

  describe "errors" do
    test "need a square tensor when dissimilarity is precomputed" do
      assert_raise(
        ArgumentError,
        "Expected a square rank 2 (`{num_obs, num_obs}`) tensor when `dissimilarity: :precomputed`, found shape: {3, 2}.",
        fn ->
          Hierarchical.fit(Nx.tensor([[1, 2], [3, 4], [5, 6]]), dissimilarity: :precomputed)
        end
      )
    end

    test "need a rank 2 tensor when dissimilarity is precomputed" do
      assert_raise(
        ArgumentError,
        "Expected a square rank 2 (`{num_obs, num_obs}`) tensor when `dissimilarity: :precomputed`, found shape: {3}.",
        fn ->
          Hierarchical.fit(Nx.tensor([1, 2, 3]), dissimilarity: :precomputed)
        end
      )
    end

    test "need a rank 2 tensor" do
      assert_raise(
        ArgumentError,
        "Expected a rank 2 (`{num_obs, num_features}`) tensor, found shape: {3}.",
        fn ->
          Hierarchical.fit(Nx.tensor([1, 2, 3]))
        end
      )
    end

    test "need at least 3 data points" do
      assert_raise(ArgumentError, "Must have a minimum of 3 data points, found: 2.", fn ->
        Hierarchical.fit(Nx.tensor([[1], [2]]))
      end)
    end

    test "num_clusters may not exceed number of data points" do
      model = Hierarchical.fit(Nx.tensor([[1], [2], [3]]))

      assert_raise(ArgumentError, "`num_clusters` may not exceed number of data points.", fn ->
        Hierarchical.labels_list(model, cluster_by: [num_clusters: 4])
      end)
    end

    test "additional option validations" do
      model = Hierarchical.fit(Nx.tensor([[1], [2], [3]]))

      assert_raise(ArgumentError, "Must pass exactly one of `:height` or `:num_clusters`", fn ->
        Hierarchical.labels_list(model, cluster_by: [num_clusters: 2, height: 1.0])
      end)
    end
  end
end
