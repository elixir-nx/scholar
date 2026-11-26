defmodule Scholar.Metrics.NeighborsTest do
  use Scholar.Case, async: true
  alias Scholar.Metrics.Neighbors
  doctest Neighbors

  describe "recall" do
    test "counts a repeated predicted index once" do
      neighbors_true = Nx.tensor([[0, 1]])

      assert_all_close(Neighbors.recall(neighbors_true, Nx.tensor([[0, 0]])), Nx.tensor(0.5))
    end

    test "never goes above 1.0 with a repeated index on both sides" do
      neighbors_true = Nx.tensor([[0, 0]])

      assert_all_close(Neighbors.recall(neighbors_true, neighbors_true), Nx.tensor(0.5))
    end

    test "averages the fraction found over the samples" do
      neighbors_true = Nx.tensor([[0, 1, 2], [3, 4, 5]])
      neighbors_pred = Nx.tensor([[0, 0, 0], [3, 4, 9]])

      assert_all_close(Neighbors.recall(neighbors_true, neighbors_pred), Nx.tensor(0.5))
    end

    test "works with jit_apply" do
      neighbors_true = Nx.tensor([[0, 1], [1, 2]])
      neighbors_pred = Nx.tensor([[0, 0], [1, 2]])

      assert_all_close(
        Nx.Defn.jit_apply(&Neighbors.recall/2, [neighbors_true, neighbors_pred]),
        Nx.tensor(0.75)
      )
    end
  end
end
