defmodule Scholar.Metrics.DiscountedCumulativeGainTest do
  use Scholar.Case, async: true
  alias Scholar.Metrics.DiscountedCumulativeGain

  describe "compute/2" do
    test "computes DCG when there are no ties" do
      y_true = Nx.tensor([3, 2, 3, 0, 1, 2])
      y_score = Nx.tensor([3.0, 2.2, 3.5, 0.5, 1.0, 2.1])

      result = DiscountedCumulativeGain.compute(y_true, y_score)

      assert %Nx.Tensor{data: data} = Nx.broadcast(result, {1})
    end

    test "computes DCG with ties" do
      y_true = Nx.tensor([3, 3, 3])
      y_score = Nx.tensor([2.0, 2.0, 3.5])

      result = DiscountedCumulativeGain.compute(y_true, y_score)

      assert %Nx.Tensor{data: data} = Nx.broadcast(result, {1})
    end

    test "raises error when shapes mismatch" do
      y_true = Nx.tensor([3, 2, 3])
      y_score = Nx.tensor([3.0, 2.2, 3.5, 0.5])

      assert_raise ArgumentError, "y_true and y_score tensors must have the same shape", fn ->
        DiscountedCumulativeGain.compute(y_true, y_score)
      end
    end

    test "computes DCG for top-k values" do
      y_true = Nx.tensor([3, 2, 3, 0, 1, 2])
      y_score = Nx.tensor([3.0, 2.2, 3.5, 0.5, 1.0, 2.1])

      result = DiscountedCumulativeGain.compute(y_true, y_score, 3)

      assert %Nx.Tensor{data: data} = Nx.broadcast(result, {1})
    end
  end
end
