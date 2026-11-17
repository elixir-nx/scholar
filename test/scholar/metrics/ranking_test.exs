defmodule Scholar.Metrics.RankingTest do
  use Scholar.Case, async: true
  alias Scholar.Metrics.Ranking
  doctest Ranking

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

    test "averages the gains of tied scores" do
      y_true = Nx.tensor([10, 0, 0, 1, 5])

      assert_all_close(
        Ranking.dcg(y_true, Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0])),
        Nx.tensor(9.435069180414054)
      )

      assert_all_close(
        Ranking.dcg(y_true, Nx.tensor([1.0, 1.0, 1.0, 0.5, 0.5])),
        Nx.tensor(9.555687274495329)
      )
    end

    test "averages a group of tied scores cut by :k" do
      y_true = Nx.tensor([10, 0, 0, 1, 5])
      y_score = Nx.tensor([1.0, 1.0, 1.0, 0.5, 0.5])

      assert_all_close(Ranking.dcg(y_true, y_score, k: 2), Nx.tensor(5.436432511904858))
    end

    test "computes DCG without truncation when :k is nil or the number of elements" do
      y_true = Nx.tensor([3, 2, 3, 0, 1, 2])
      y_score = Nx.tensor([3.0, 2.2, 3.5, 0.5, 1.0, 2.1])

      assert Ranking.dcg(y_true, y_score, k: nil) == Ranking.dcg(y_true, y_score)
      assert Ranking.dcg(y_true, y_score, k: 6) == Ranking.dcg(y_true, y_score)
    end

    test "computes DCG with more than 255 groups of tied scores" do
      y = Nx.iota({300}, type: :f64)

      assert_all_close(Ranking.dcg(y, y), Nx.tensor(8137.175360171257, type: :f64))
    end

    test "keeps f64 precision" do
      y_true = Nx.tensor([10, 0, 0, 1, 5], type: :f64)
      y_score = Nx.tensor([0.1, 0.2, 0.3, 4.0, 70.0], type: :f64)

      result = Ranking.dcg(y_true, y_score)

      assert Nx.type(result) == {:f, 64}
      assert_all_close(result, Nx.tensor(9.499457825916874, type: :f64), atol: 1.0e-14)
    end

    test "raises when :k is not a positive integer" do
      y_true = Nx.tensor([3, 2, 3])
      y_score = Nx.tensor([3.0, 2.2, 3.5])

      assert_raise NimbleOptions.ValidationError, ~r/expected positive integer, got: 0.5/, fn ->
        Ranking.dcg(y_true, y_score, k: 0.5)
      end
    end

    test "works with jit_apply" do
      y_true = Nx.tensor([10, 0, 0, 1, 5])
      y_score = Nx.tensor([1.0, 1.0, 1.0, 0.5, 0.5])

      result = Nx.Defn.jit_apply(&Ranking.dcg/3, [y_true, y_score, [k: 2]])

      assert_all_close(result, Ranking.dcg(y_true, y_score, k: 2))
    end
  end

  describe "ndcg_n/3" do
    test "averages the gains of tied scores" do
      y_true = Nx.tensor([10, 0, 0, 1, 5])

      assert_all_close(
        Ranking.ndcg_n(y_true, Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0])),
        Nx.tensor(0.6909785334518438)
      )

      assert_all_close(
        Ranking.ndcg_n(y_true, Nx.tensor([1.0, 1.0, 1.0, 0.5, 0.5])),
        Nx.tensor(0.6998120154499459)
      )
    end

    test "raises on invalid options" do
      y_true = Nx.tensor([3, 2, 3])
      y_score = Nx.tensor([3.0, 2.2, 3.5])

      assert_raise NimbleOptions.ValidationError, ~r/expected positive integer, got: -1/, fn ->
        Ranking.ndcg_n(y_true, y_score, k: -1)
      end

      assert_raise NimbleOptions.ValidationError, ~r/unknown options \[:foo\]/, fn ->
        Ranking.ndcg_n(y_true, y_score, foo: 1)
      end
    end
  end
end
