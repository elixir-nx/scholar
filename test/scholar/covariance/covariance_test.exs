defmodule Scholar.CovarianceTest do
  use Scholar.Case, async: true
  alias Scholar.Covariance
  doctest Covariance

  describe "errors" do
    test "rank of input not equal to 2" do
      assert_raise ArgumentError,
                   "expected data to have rank equal 2, got: 3",
                   fn -> Covariance.covariance_matrix(Nx.tensor([[[1, 2], [3, 4]]])) end
    end
  end
end
