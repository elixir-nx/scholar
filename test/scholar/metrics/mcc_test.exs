defmodule Scholar.Metrics.MCCTest do
  use ExUnit.Case, async: true
  alias Scholar.Metrics.MCC

  describe "MCC.compute/2" do
    test "returns 1 for perfect predictions" do
      y_true = Nx.tensor([1, 0, 1, 0, 1])
      y_pred = Nx.tensor([1, 0, 1, 0, 1])
      assert MCC.compute(y_true, y_pred) == Nx.tensor([1.0], type: :f32)
    end

    test "returns -1 for completely wrong predictions" do
      y_true = Nx.tensor([1, 0, 1, 0, 1])
      y_pred = Nx.tensor([0, 1, 0, 1, 0])
      assert MCC.compute(y_true, y_pred) == Nx.tensor([-1.0], type: :f32)
    end

    test "returns 0 when all predictions are positive" do
      y_true = Nx.tensor([1, 0, 1, 0, 1])
      y_pred = Nx.tensor([1, 1, 1, 1, 1])
      assert MCC.compute(y_true, y_pred) == Nx.tensor([0.0], type: :f32)
    end

    test "returns 0 when all predictions are negative" do
      y_true = Nx.tensor([1, 0, 1, 0, 1])
      y_pred = Nx.tensor([0, 0, 0, 0, 0])
      assert MCC.compute(y_true, y_pred) == Nx.tensor([0.0], type: :f32)
    end

    test "computes MCC for generic case" do
      y_true = Nx.tensor([1, 0, 1, 0, 1])
      y_pred = Nx.tensor([1, 0, 1, 1, 1])
      assert MCC.compute(y_true, y_pred) == Nx.tensor([0.6123723983764648], type: :f32)
    end

    test "returns 0 when TP, TN, FP, and FN are all 0" do
      y_true = Nx.tensor([0, 0, 0, 0, 0])
      y_pred = Nx.tensor([0, 0, 0, 0, 0])
      assert MCC.compute(y_true, y_pred) == Nx.tensor([0.0], type: :f32)
    end
  end
end
