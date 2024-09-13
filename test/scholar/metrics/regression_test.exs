defmodule Scholar.Metrics.RegressionTest do
  use Scholar.Case, async: true

  alias Scholar.Metrics.Regression
  doctest Regression

  describe "mean_tweedie_deviance!/3" do
    test "raise when y_pred <= 0 and power < 0" do
      power = -1
      y_true = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :u32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], type: :u32)

      assert_raise RuntimeError, ~r/mean Tweedie deviance/, fn ->
        Regression.mean_tweedie_deviance!(y_true, y_pred, power)
      end
    end

    test "raise when y_pred <= 0 and 1 <= power < 2" do
      power = 1
      y_true = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :u32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], type: :u32)

      assert_raise RuntimeError, ~r/mean Tweedie deviance/, fn ->
        Regression.mean_tweedie_deviance!(y_true, y_pred, power)
      end
    end

    test "raise when y_pred <= 0 and power >= 2" do
      power = 2
      y_true = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :u32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], type: :u32)

      assert_raise RuntimeError, ~r/mean Tweedie deviance/, fn ->
        Regression.mean_tweedie_deviance!(y_true, y_pred, power)
      end
    end

    test "raise when y_true < 0 and 1 <= power < 2" do
      power = 1
      y_true = Nx.tensor([-1, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :s32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], type: :s32)

      assert_raise RuntimeError, ~r/mean Tweedie deviance/, fn ->
        Regression.mean_tweedie_deviance!(y_true, y_pred, power)
      end
    end

    test "raise when y_true <= 0 and power >= 2" do
      power = 2
      y_true = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :s32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], type: :s32)

      assert_raise RuntimeError, ~r/mean Tweedie deviance/, fn ->
        Regression.mean_tweedie_deviance!(y_true, y_pred, power)
      end
    end
  end

  describe "d2_tweedie_score/3" do
    test "equal R^2 when power is 0" do
      y_true = Nx.tensor([1, 1, 1, 1, 1, 2, 2, 1, 3, 1], type: :u32)
      y_pred = Nx.tensor([2, 2, 1, 1, 2, 2, 2, 1, 3, 1], type: :u32)
      d2 = Regression.d2_tweedie_score(y_true, y_pred, 0)
      r2 = Regression.r2_score(y_true, y_pred)

      assert Nx.equal(d2, r2)
    end
  end

  describe "mean_pinball_loss/3" do
    test "mean_pinball_loss cases from sklearn" do
      # Test cases copied from sklearn:
      # https://github.com/scikit-learn/scikit-learn/blob/128e40ed593c57e8b9e57a4109928d58fa8bf359/sklearn/metrics/tests/test_regression.py#L49      

      y_true = Nx.linspace(1, 50, n: 50)
      y_pred = Nx.add(y_true, 1)
      y_pred_2 = Nx.add(y_true, -1)

      assert Regression.mean_pinball_loss(y_true, y_pred) == Nx.tensor(0.5)
      assert Regression.mean_pinball_loss(y_true, y_pred_2) == Nx.tensor(0.5)
      assert Regression.mean_pinball_loss(y_true, y_pred, alpha: 0.4) == Nx.tensor(0.6)
      assert Regression.mean_pinball_loss(y_true, y_pred_2, alpha: 0.4) == Nx.tensor(0.4)
    end

    test "mean_pinball_loss with axes" do
      y_true = Nx.tensor([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
      y_pred = Nx.tensor([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])

      sample_weight =
        Nx.tensor([[0.5, 0.5, 0.5, 1.5], [1.5, 0.5, 1.5, 1.5], [1.5, 1.5, 1.5, 1.5]])

      expected_error = Nx.tensor((1 + 2 / 3) / 8)
      expected_raw_values_tensor = Nx.tensor([0.5, 0.33333333, 0.0, 0.0])
      expected_raw_values_weighted_tensor = Nx.tensor([0.5, 0.4, 0.0, 0.0])

      mpbl = Regression.mean_pinball_loss(y_true, y_pred)
      assert_all_close(mpbl, expected_error)
      ## this assertion yields false due to precision error
      mpbl =
        Regression.mean_pinball_loss(
          y_true,
          y_pred,
          alpha: 0.5
        )

      assert_all_close(mpbl, expected_error)
      mpbl = Regression.mean_pinball_loss(y_true, y_pred, alpha: 0.5, axes: [0])
      assert_all_close(mpbl, expected_raw_values_tensor)

      mpbl =
        Regression.mean_pinball_loss(y_true, y_pred,
          alpha: 0.5,
          sample_weights: sample_weight,
          axes: [0]
        )

      assert_all_close(mpbl, expected_raw_values_weighted_tensor)

      mpbl =
        Regression.mean_pinball_loss(y_true, y_pred,
          alpha: 0.5,
          sample_weights: sample_weight
        )

      assert_all_close(mpbl, Nx.tensor(0.225))
    end
  end
end
