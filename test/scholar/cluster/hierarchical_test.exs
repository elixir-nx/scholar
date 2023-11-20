defmodule Scholar.Cluster.HierarchicalTest do
  use Scholar.Case, async: true
  alias Scholar.Cluster.Hierarchical
  # doctest Hierarchical

  describe "fit" do
    test "a basic example" do
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
      # points as singleton clusters. The first step of the algorithm merges singleton clusters
      # 0: [0] and 1: [1] to form cluster 9: [0, 1]. This process continues until all clusters have
      # been merged into a single cluster with all points.
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
      result =
        Hierarchical.fit(data,
          dissimilarity: :euclidean,
          group_by: [num_clusters: 3],
          linkage: :single
        )

      # The dendrogram formation part of the algorithm should've formed the following clusters,
      # dissimilarities, and sizes (which collectively form the dendrogram).
      assert result.clusters ==
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

      assert result.dissimilarities == Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
      assert result.sizes == Nx.tensor([2, 2, 2, 3, 3, 3, 6, 9])

      # The clustering part of the algorithm uses the `group_by: [num_clusters: 3]` option to take
      # the dendrogram and form 3 clusters. This should result in each datum having the following
      # cluster labels.
      assert result.labels == Nx.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    end

    test "group_by option not required" do
      result =
        Hierarchical.fit(Nx.tensor([[2], [7], [9], [0], [3]]),
          dissimilarity: :euclidean,
          linkage: :single
        )

      assert result.clusters == Nx.tensor([[0, 4], [1, 2], [3, 5], [6, 7]])
      assert result.dissimilarities == Nx.tensor([1.0, 2.0, 2.0, 4.0])
      assert result.sizes == Nx.tensor([2, 2, 3, 5])

      # Not providing the `group_by` option results in no labels.
      assert result.labels == nil
    end
  end

  describe "errors" do
    test "num_clusters may not exceed number of datapoints" do
      assert_raise(ArgumentError, "`num_clusters` may not exceed number of data points", fn ->
        Hierarchical.fit(Nx.tensor([[2], [7], [9], [0], [3]]),
          dissimilarity: :euclidean,
          group_by: [num_clusters: 6],
          linkage: :single
        )
      end)
    end
  end
end
