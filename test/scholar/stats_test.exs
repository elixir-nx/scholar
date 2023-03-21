defmodule Scholar.StatsTest do
  use Scholar.Case, async: true
  alias Scholar.Stats
  doctest Stats

  defp x do
    Nx.tensor([[3, 5, 3], [2, 6, 1], [9, 3, 2], [1, 6, 8]])
  end

  describe "moment" do
    test "moment set to 3" do
      expected_moment = Nx.tensor([29.53125, -1.5, 18.0])
      assert_all_close(Stats.moment(x(), 3), expected_moment)
    end

    test "moment set to 3 and axis set to 1" do
      expected_moment = Nx.tensor([0.5925924181938171, 6.0, 19.25926399230957, -12.0])
      assert_all_close(Stats.moment(x(), 3, axes: [1]), expected_moment)
    end

    test "moment set to 3 and axis set to 1 and keep_axes set to true" do
      expected_moment = Nx.tensor([[0.5925924181938171], [6.0], [19.25926399230957], [-12.0]])
      assert_all_close(Stats.moment(x(), 3, axes: [1], keep_axes: true), expected_moment)
    end

    test "moment set to 3 and axis set to nil" do
      expected_moment = Nx.tensor(9.438657407407414)
      assert_all_close(Stats.moment(x(), 3, axes: nil), expected_moment)
    end

    test "moment set to 3 and axis set to nil and keep_axes set to true" do
      expected_moment = Nx.tensor([[9.438657407407414]])
      assert_all_close(Stats.moment(x(), 3, axes: nil, keep_axes: true), expected_moment)
    end

    test "moment set to 3 and axis set to [0, 1] and keep_axes set to true" do
      expected_moment = Nx.tensor([[9.438657407407414]])
      assert_all_close(Stats.moment(x(), 3, axes: [0, 1], keep_axes: true), expected_moment)
    end
  end

  describe "skew" do
    test "all defaults" do
      expected_skew = Nx.tensor([0.97940938, -0.81649658, 0.9220734])
      assert_all_close(Stats.skew(x()), expected_skew)
    end

    test "axis set to 1" do
      expected_skew = Nx.tensor([0.70710678, 0.59517006, 0.65201212, -0.47033046])
      assert_all_close(Stats.skew(x(), axes: [1]), expected_skew)
    end

    test "axis set to 1 and keep_axes set to true" do
      expected_skew = Nx.tensor([[0.70710678], [0.59517006], [0.65201212], [-0.47033046]])
      assert_all_close(Stats.skew(x(), axes: [1], keep_axes: true), expected_skew)
    end

    test "axis set to nil" do
      expected_skew = Nx.tensor(0.5596660882003394)
      assert_all_close(Stats.skew(x(), axes: nil), expected_skew)
    end

    test "axis set to nil and keep_axes set to true" do
      expected_skew = Nx.tensor([[0.5596660882003394]])
      assert_all_close(Stats.skew(x(), axes: nil, keep_axes: true), expected_skew)
    end

    test "axis set to [0, 1] and keep_axes set to true" do
      expected_skew = Nx.tensor([[0.5596660882003394]])
      assert_all_close(Stats.skew(x(), axes: [0, 1], keep_axes: true), expected_skew)
    end

    test "axis set to [1] and bias set to false" do
      expected_skew = Nx.tensor([1.73205081, 1.45786297, 1.59709699, -1.15206964])
      assert_all_close(Stats.skew(x(), axes: [1], bias: false), expected_skew)
    end
  end

  describe "kurtosis" do
    test "all defaults" do
      expected_kurtosis = Nx.tensor([-0.79808533, -1.0, -0.83947681])
      assert_all_close(Stats.kurtosis(x()), expected_kurtosis)
    end

    test "axis set to 1" do
      expected_kurtosis = Nx.tensor([-1.5, -1.5, -1.5, -1.5])
      assert_all_close(Stats.kurtosis(x(), axes: [1]), expected_kurtosis)
    end

    test "axis set to 1 and keep_axes set to true" do
      expected_kurtosis = Nx.tensor([[-1.5], [-1.5], [-1.5], [-1.5]])
      assert_all_close(Stats.kurtosis(x(), axes: [1], keep_axes: true), expected_kurtosis)
    end

    test "axis set to nil" do
      expected_kurtosis = Nx.tensor(-0.9383737228328437)
      assert_all_close(Stats.kurtosis(x(), axes: nil), expected_kurtosis)
    end

    test "axis set to nil and keep_axes set to true" do
      expected_kurtosis = Nx.tensor([[-0.9383737228328437]])
      assert_all_close(Stats.kurtosis(x(), axes: nil, keep_axes: true), expected_kurtosis)
    end

    test "axis set to [0, 1] and keep_axes set to true" do
      expected_kurtosis = Nx.tensor([[-0.9383737228328437]])
      assert_all_close(Stats.kurtosis(x(), axes: [0, 1], keep_axes: true), expected_kurtosis)
    end

    test "axis set to nil and bias set to false" do
      expected_kurtosis = Nx.tensor(-0.757638248501074)
      assert_all_close(Stats.kurtosis(x(), axes: nil, bias: false), expected_kurtosis)
    end

    test "axis set to nil and bias set to false and variant set to pearson" do
      expected_kurtosis = Nx.tensor(2.242361751498926)

      assert_all_close(
        Stats.kurtosis(x(), axes: nil, bias: false, variant: :pearson),
        expected_kurtosis
      )
    end
  end
end
