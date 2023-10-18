defmodule Scholar.Metrics.RegressionTest do
  use Scholar.Case, async: true

  alias Scholar.Metrics.Regression
  doctest Regression

  describe "mean_tweedie_deviance" do
    test "raise when y_pred <= 0 and power < 0 with check_tensors: true" do
      power = -1
      y_true = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :u32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], type: :u32)

      assert_raise RuntimeError, ~r/Mean Tweedie deviance/, fn ->
        Regression.mean_tweedie_deviance(y_true, y_pred, power, check_tensors: true)
      end
    end

    test "raise when y_pred <= 0 and 1 <= power < 2 with check_tensors: true" do
      power = 1
      y_true = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :u32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], type: :u32)

      assert_raise RuntimeError, ~r/Mean Tweedie deviance/, fn ->
        Regression.mean_tweedie_deviance(y_true, y_pred, power, check_tensors: true)
      end
    end

    test "raise when y_pred <= 0 and power >= 2 with check_tensors: true" do
      power = 2
      y_true = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :u32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], type: :u32)

      assert_raise RuntimeError, ~r/Mean Tweedie deviance/, fn ->
        Regression.mean_tweedie_deviance(y_true, y_pred, power, check_tensors: true)
      end
    end

    test "raise when y_true < 0 and 1 <= power < 2 with check_tensors: true" do
      power = 1
      y_true = Nx.tensor([-1, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :s32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], type: :s32)

      assert_raise RuntimeError, ~r/Mean Tweedie deviance/, fn ->
        Regression.mean_tweedie_deviance(y_true, y_pred, power, check_tensors: true)
      end
    end

    test "raise when y_true <= 0 and power >= 2 with check_tensors: true" do
      power = 2
      y_true = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :s32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], type: :s32)

      assert_raise RuntimeError, ~r/Mean Tweedie deviance/, fn ->
        Regression.mean_tweedie_deviance(y_true, y_pred, power, check_tensors: true)
      end
    end
  end
end
