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

    test "returns similarity when tensors have a single element" do
      x = Nx.tensor([1])
      y = Nx.tensor([2])

      assert Similarity.jaccard(x, y) == Nx.tensor(0.0)
    end

    test "returns similarity when tensor has multiple dimensions" do
      x = Nx.tensor([[0, 1, 1], [1, 1, 0]])
      y = Nx.tensor([[1, 1, 1], [1, 0, 0]])

      assert Similarity.jaccard(x, y) == Nx.tensor(0.5)
    end

    test "raises exception when tensors have different shapes" do
      x = Nx.tensor([1, 2, 3, 5])
      y = Nx.tensor([1, 30, 4, 8, 9])

      assert_raise ArgumentError,
                   "expected tensor to have shape {4}, got tensor with shape {5}",
                   fn ->
                     Similarity.jaccard(x, y)
                   end
    end

    test "raises exception when tensors have shape zero" do
      x = Nx.tensor(1)
      y = Nx.tensor(1)

      assert_raise RuntimeError, "expected input shape of at least {1}, got: {}", fn ->
        Similarity.jaccard(x, y)
      end
    end
  end

  describe "binary_jaccard/2" do
    test "returns similarity according to sklearn jaccard_score function" do
      x = Nx.tensor([1, 0, 0, 1, 1, 1])
      y = Nx.tensor([0, 0, 1, 1, 1, 0])

      assert Similarity.binary_jaccard(x, y) == Nx.tensor(0.4000000059604645)
    end

    test "returns 100% of similarity" do
      x = Nx.tensor([1, 0, 1])
      y = Nx.tensor([1, 0, 1])

      assert Similarity.binary_jaccard(x, y) == Nx.tensor(1.0)
    end

    test "returns 0% of similarity" do
      x = Nx.tensor([1, 1, 1])
      y = Nx.tensor([0, 0, 0])

      assert Similarity.binary_jaccard(x, y) == Nx.tensor(0.0)
    end

    test "returns 20% of similarity" do
      x = Nx.tensor([1, 0, 1, 0, 1])
      y = Nx.tensor([0, 1, 1, 1, 0])

      assert Similarity.binary_jaccard(x, y) == Nx.tensor(0.20000000298023224)
    end

    test "returns similarity when tensors have a single element" do
      x = Nx.tensor([1])
      y = Nx.tensor([1])

      assert Similarity.binary_jaccard(x, y) == Nx.tensor(1.0)
    end

    test "returns similarity when tensor has multiple dimensions" do
      x = Nx.tensor([[0, 1, 1], [1, 1, 1]])
      y = Nx.tensor([[1, 1, 1], [1, 1, 1]])

      assert Similarity.binary_jaccard(x, y) == Nx.tensor(0.8333333134651184)
    end

    test "raises exception when tensors have different shapes" do
      x = Nx.tensor([1, 1, 0])
      y = Nx.tensor([1, 0])

      assert_raise ArgumentError,
                   "expected tensor to have shape {3}, got tensor with shape {2}",
                   fn ->
                     Similarity.binary_jaccard(x, y)
                   end
    end

    test "raises exception when tensors have shape zero" do
      x = Nx.tensor(1)
      y = Nx.tensor(1)

      assert_raise RuntimeError, "expected input shape of at least {1}, got: {}", fn ->
        Similarity.jaccard(x, y)
      end
    end
  end
end
