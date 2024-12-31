defmodule Scholar.Preprocessing.RobustScalerTest do
  use Scholar.Case, async: true
  alias Scholar.Preprocessing.RobustScaler
  doctest RobustScaler

  describe "fit_transform" do
    test "applies scaling to data" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])

      expected =
        Nx.tensor([
          [0.0, -1.0, 1.3333333333333333],
          [1.0, 0.0, 0.0],
          [-1.0, 1.0, -0.6666666666666666]
        ])

      assert_all_close(RobustScaler.fit_transform(data), expected)
    end

    test "applies scaling to data with custom quantile range" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])

      expected =
        Nx.tensor([
          [0.0, -0.7142857142857142, 1.0],
          [0.7142857142857142, 0.0, 0.0],
          [-0.7142857142857142, 0.7142857142857142, -0.5]
        ])

      assert_all_close(
        RobustScaler.fit_transform(data, quantile_range: {10, 80}),
        expected
      )
    end

    test "handles constant data (all values the same)" do
      data = Nx.tensor([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
      expected = Nx.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

      assert_all_close(RobustScaler.fit_transform(data), expected)
    end

    test "handles already scaled data" do
      data = Nx.tensor([[0, -1, 1], [1, 0, 0], [-1, 1, -1]])
      expected = data

      assert_all_close(RobustScaler.fit_transform(data), expected)
    end

    test "handles single-row tensor" do
      data = Nx.tensor([[1, 2, 3]])
      expected = Nx.tensor([[0.0, 0.0, 0.0]])

      assert_all_close(RobustScaler.fit_transform(data), expected)
    end

    test "handles single-column tensor" do
      data = Nx.tensor([[1], [2], [3]])
      expected = Nx.tensor([[-1.0], [0.0], [1.0]])

      assert_all_close(RobustScaler.fit_transform(data), expected)
    end

    test "handles data with negative values only" do
      data = Nx.tensor([[-5, -10, -15], [-15, -5, -20], [-10, -15, -5]])

      expected =
        Nx.tensor([
          [1.0, 0.0, 0.0],
          [-1.0, 1.0, -0.6666666666666666],
          [0.0, -1.0, 1.3333333333333333]
        ])

      assert_all_close(RobustScaler.fit_transform(data), expected)
    end

    test "handles data with extreme outliers" do
      data = Nx.tensor([[1, 2, 3], [1000, 2000, 3000], [-1000, -2000, -3000]])

      expected =
        Nx.tensor([[0.0, 0.0, 0.0], [0.999, 0.999, 0.999], [-1.001, -1.001, -1.001]])

      assert_all_close(
        RobustScaler.fit_transform(data),
        expected
      )
    end
  end

  describe "errors" do
    test "wrong input rank for fit" do
      assert_raise ArgumentError,
                   "expected tensor to have shape {num_samples, num_features}, got tensor with shape: {1, 1, 1}",
                   fn ->
                     RobustScaler.fit(Nx.tensor([[[1]]]))
                   end
    end

    test "wrong input rank for transform" do
      assert_raise ArgumentError,
                   "expected tensor to have shape {num_samples, num_features}, got tensor with shape: {1, 1, 1}",
                   fn ->
                     RobustScaler.fit(Nx.tensor([[1]]))
                     |> RobustScaler.transform(Nx.tensor([[[1]]]))
                   end
    end

    test "wrong quantile range" do
      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :quantile_range option: expected :quantile_range to be a tuple {q_min, q_max} such that 0.0 < q_min < q_max < 100.0, got: {10, 800}",
                   fn ->
                     RobustScaler.fit(Nx.tensor([[[1]]]), quantile_range: {10, 800})
                   end
    end
  end
end
