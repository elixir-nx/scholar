defmodule Scholar.Metrics.RankingTest do
  use Scholar.Case, async: true
  alias Scholar.Metrics.Ranking

  describe "dcg/3" do
    test "computes DCG when there are no ties" do
      y_true = Nx.tensor([3, 2, 3, 0, 1, 2])
      y_score = Nx.tensor([3.0, 2.2, 3.5, 0.5, 1.0, 2.1])

      result = Ranking.dcg(y_true, y_score)

      x = Nx.tensor([7.140995025634766])
      assert x == Nx.broadcast(result, {1})
    end

    test "computes DCG with ties" do
      y_true = Nx.tensor([3, 3, 3])
      y_score = Nx.tensor([2.0, 2.0, 3.5])

      result = Ranking.dcg(y_true, y_score)

      x = Nx.tensor([6.3927892607143715])
      assert x == Nx.broadcast(result, {1})
    end

    test "raises error when shapes mismatch" do
      y_true = Nx.tensor([3, 2, 3])
      y_score = Nx.tensor([3.0, 2.2, 3.5, 0.5])

      assert_raise ArgumentError,
                   "expected tensor to have shape {3}, got tensor with shape {4}",
                   fn ->
                     Ranking.dcg(y_true, y_score)
                   end
    end

    test "computes DCG for top-k values" do
      y_true = Nx.tensor([3, 2, 3, 0, 1, 2])
      y_score = Nx.tensor([3.0, 2.2, 3.5, 0.5, 1.0, 2.1])

      result = Ranking.dcg(y_true, y_score, k: 3)

      x = Nx.tensor([5.892789363861084])
      assert x == Nx.broadcast(result, {1})
    end
  end
end
