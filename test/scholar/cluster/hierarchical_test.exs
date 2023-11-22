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
      # points as singleton clades. The first step of the algorithm merges singleton clades
      # 0: [0] and 1: [1] to form clade 9: [0, 1]. This process continues until all clades have
      # been merged into a single clade with all points.
      #
      #       0   1   2   3   4   5   6   7   8
      #    8: [0] [1] [2] [3] [4] [5] [6] [7] [8]
      #       9    2   3   4   5   6   7   8
      #    9: [01] [2] [3] [4] [5] [6] [7] [8]
      #       ----
      #       9    2   10   5   6   7   8
      #   10: [01] [2] [34] [5] [6] [7] [8]
      #                ----
      #       9    2   10   5   11   8
      #   11: [01] [2] [34] [5] [67] [8]
      #                         ----
      #       12    10   5   11   8
      #   12: [012] [34] [5] [67] [8]
      #       -----
      #       12    13    11   8
      #   13: [012] [345] [67] [8]
      #             -----
      #       12    13    14
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
                 [3, 4],
                 [6, 7],
                 [2, 9],
                 [5, 10],
                 [8, 11],
                 [12, 13],
                 [14, 15]
               ])

      assert model.dissimilarities ==
               Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])

      assert model.sizes == Nx.tensor([2, 2, 2, 3, 3, 3, 6, 9])

      # The clustering part of the algorithm uses the `cluster_by: [num_clusters: 3]` option to
      # take the model and form 3 clusters. This should result in each datum having the following
      # cluster labels.
      clusters = Hierarchical.fit_predict(model, cluster_by: [num_clusters: 3])
      assert clusters == Nx.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    end
  end

  @data Nx.tensor([[2], [7], [9], [0], [3]])

  describe "fit_predict" do
    test "cluster by height" do
      clusters = Hierarchical.fit_predict(@data, cluster_by: [height: 2.5])
      assert clusters == Nx.tensor([0, 1, 1, 0, 0])
    end

    test "cluster by number of clusters" do
      clusters = Hierarchical.fit_predict(@data, cluster_by: [num_clusters: 3])
      assert clusters == Nx.tensor([0, 1, 1, 2, 0])
    end

    test "works with model" do
      model = Hierarchical.fit(@data)
      clusters = Hierarchical.fit_predict(model, cluster_by: [height: 2.5])
      assert clusters == Nx.tensor([0, 1, 1, 0, 0])
    end
  end

  describe "errors" do
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
      model = Hierarchical.fit(@data)

      assert_raise(ArgumentError, "`num_clusters` may not exceed number of data points.", fn ->
        Hierarchical.fit_predict(model, cluster_by: [num_clusters: 6])
      end)
    end
  end
end
