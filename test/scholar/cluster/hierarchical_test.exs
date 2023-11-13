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
          # group_by: [num_clusters: 3],
          group_by: [height: 2.0],
          linkage: :single
        )

      assert result.dendrogram == [
               {9, [1, 0], 1.0},
               {10, [4, 3], 1.0},
               {11, [7, 6], 1.0},
               {12, [9, 2], 1.0},
               {13, [10, 5], 1.0},
               {14, [11, 8], 1.0},
               {15, [13, 12], 2.0},
               {16, [15, 14], 2.0}
             ]

      assert result.labels == Nx.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

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
          linkage: :centroid
        )

      dendrogram
      |> IO.inspect(label: :final_dendrogram)
    end
  end

  describe "experiments" do
    test "nearest neighbor chain" do
      # data = Nx.tensor([[1, 5], [2, 5], [1, 4], [4, 5], [5, 5], [5, 4], [1, 2], [1, 1], [2, 1]])
      data = Nx.tensor([[2], [7], [9], [0], [3]])
      # data = Nx.tensor([[1], [1], [1], [1], [1], [1], [1], [10], [50]])
      # {n_row, _} = Nx.shape(data)
      # nan_diag = Nx.Constants.nan() |> Nx.broadcast({n_row}) |> Nx.make_diagonal()
      # pairwise = data |> Hierarchical.pairwise_euclidean() |> Nx.add(nan_diag) |> IO.inspect()

      # Hierarchical.link(pairwise)
      # |> IO.inspect()

      Hierarchical.fit(data,
        dissimilarity: :euclidean,
        # group_by: [num_clusters: 3],
        # group_by: [height: 2.0],
        linkage: :single
      )
      |> IO.inspect(charlists: :as_list)
    end
  end

  describe "condensed matrix" do
    test "has the correct size" do
      for n <- 2..10 do
        size =
          {n, n}
          |> Nx.iota()
          |> Hierarchical.CondensedMatrix.condense_pairwise()
          |> Nx.size()
          |> Nx.tensor()

        assert size == Hierarchical.CondensedMatrix.tri(n - 1)
      end
    end

    test "can be indexed correctly" do
      for n <- 2..10 do
        rcs = Hierarchical.CondensedMatrix.pairwise_indices(n)
        is = Hierarchical.CondensedMatrix.rc_to_i(rcs)
        assert is == Nx.iota({Nx.to_number(Hierarchical.CondensedMatrix.tri(n - 1))})
      end
    end
  end
end
