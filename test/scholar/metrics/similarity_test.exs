defmodule Scholar.Metrics.SimilarityTest do
  use ExUnit.Case
  alias Scholar.Metrics.Similarity
  doctest Similarity

  describe "jaccard/2" do
    test "returns similarity according to sklearn jaccard_score function" do
      x = Nx.tensor([1, 2, 3, 5, 0])
      y = Nx.tensor([1, 30, 4, 8, 9])

      assert Similarity.jaccard(x, y) == Nx.tensor(0.1111111119389534)
    end

    test "returns 100% of similarity" do
      x = Nx.tensor([1, 2, 3])
      y = Nx.tensor([1, 2, 3])

      assert Similarity.jaccard(x, y) == Nx.tensor(1.0)
    end

    test "returns 0% of similarity" do
      x = Nx.tensor([1, 2, 3])
      y = Nx.tensor([4, 5, 6])

      assert Similarity.jaccard(x, y) == Nx.tensor(0.0)
    end

    test "returns 20% of similarity" do
      x = Nx.tensor([1, 2, 3])
      y = Nx.tensor([3, 4, 5])

      assert Similarity.jaccard(x, y) == Nx.tensor(0.20)
    end

    test "raises exception when tensors have different shapes" do
      x = Nx.tensor([1, 2, 3, 5])
      y = Nx.tensor([1, 30, 4, 8, 9])

      assert_raise ArgumentError, "expected input shapes to be equal, got {4} != {5}", fn ->
        Similarity.jaccard(x, y)
      end
    end
  end
end
