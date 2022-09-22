defmodule Scholar.CovarianceTest do
  use ExUnit.Case, async: true

  doctest Scholar.Covariance

  @x Nx.tensor([[8, 3, 6], [2, 4, 1], [4, 5, 8]])

  test "all defaults" do
    assert Scholar.Covariance.covariance_matrix(@x) ==
             Nx.tensor([
               [6.222221851348877, -1.3333333730697632, 4.0],
               [-1.3333333730697632, 0.6666666865348816, 0.6666666865348816],
               [4.0, 0.6666666865348816, 8.666666984558105]
             ])
  end

  test "biased set to false" do
    assert Scholar.Covariance.covariance_matrix(@x, biased: false) ==
             Nx.tensor([
               [9.333333015441895, -2.0, 6.0],
               [-2.0, 1.0, 1.0],
               [6.0, 1.0, 13.0]
             ])
  end

  describe "errors" do
    test "rank of input not equal to 2" do
      assert_raise ArgumentError,
                   "expected data to have rank equal 2, got: 3",
                   fn -> Scholar.Covariance.covariance_matrix(Nx.tensor([[[1, 2], [3, 4]]])) end
    end
  end
end
