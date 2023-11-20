defmodule Scholar.Cluster.HierarchicalTest do
  use Scholar.Case, async: true
  alias Scholar.Cluster.Hierarchical
  # doctest Hierarchical

  describe "fit" do
    test "works" do
      # 5 | 0 1   3 4
      # 4 | 2       5
      # 3 |
      # 2 | 6
      # 1 | 7 8
      # 0 +-+-+-+-+-+
      #   0 1 2 3 4 5
      data = Nx.tensor([[1, 5], [2, 5], [1, 4], [4, 5], [5, 5], [5, 4], [1, 2], [1, 1], [2, 1]])

      result =
        Hierarchical.fit(data,
          dissimilarity: :euclidean,
          group_by: [num_clusters: 3],
          linkage: :single
        )

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
      assert result.labels == Nx.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
      assert result.sizes == Nx.tensor([2, 2, 2, 3, 3, 3, 6, 9])

      #  8: [0] [1] [2] [3] [4] [5] [6] [7] [8]
      #  9: [01] [2] [3] [4] [5] [6] [7] [8]
      #      --
      # 10: [01] [2] [34] [5] [6] [7] [8]
      #               --
      # 11: [01] [2] [34] [5] [67] [8]
      #                        --
      # 12: [012] [34] [5] [67] [8]
      #      ---
      # 13: [012] [345] [67] [8]
      #            ---
      # 14: [012] [345] [678]
      #                  ---
      # 15: [012345] [678]
      #      ------
      # 16: [012345678]
      #      ---------
    end

    test "generic" do
      data = Nx.tensor([[1, 5], [2, 5], [1, 4], [4, 5], [5, 5]])

      %{dendrogram: dendrogram} =
        Hierarchical.fit(data,
          dissimilarity: :euclidean,
          linkage: :single
        )

      dendrogram
      |> IO.inspect(label: :final_dendrogram)
    end
  end

  describe "experiments" do
    test "nearest neighbor chain" do
      data = Nx.tensor([[2], [7], [9], [0], [3]])

      Hierarchical.fit(data,
        dissimilarity: :euclidean,
        linkage: :single
      )
      |> IO.inspect()
    end
  end
end
