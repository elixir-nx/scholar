defmodule Scholar.Preprocessing.MinMaxScalerTest do
  use Scholar.Case, async: true
  alias Scholar.Preprocessing.MinMaxScaler

  doctest MinMaxScaler

  describe "fit_transform/2" do
    test "set axes to [0]" do
      data = Nx.tensor([[1, -1, 2], [3, 0, 0], [0, 1, -1], [2, 3, 1]])

      expected =
        Nx.tensor([
          [0.3333333432674408, 0.0, 1.0],
          [1.0, 0.25, 0.3333333432674408],
          [0.0, 0.5, 0.0],
          [0.6666666865348816, 1.0, 0.6666666865348816]
        ])

      assert_all_close(MinMaxScaler.fit_transform(data, axes: [0]), expected)
    end

    test "set axes to [0], min_bound to 1, and max_bound to 3" do
      data = Nx.tensor([[1, -1, 2], [3, 0, 0], [0, 1, -1], [2, 3, 1]])

      expected =
        Nx.tensor([
          [1.6666667461395264, 1.0, 3.0],
          [3.0, 1.5, 1.6666667461395264],
          [1.0, 2.0, 1.0],
          [2.3333334922790527, 3.0, 2.3333334922790527]
        ])

      assert_all_close(
        MinMaxScaler.fit_transform(data, axes: [0], min_bound: 1, max_bound: 3),
        expected
      )
    end

    test "Work in case where tensor contains only zeros" do
      data = Nx.broadcast(Nx.f32(0), {3, 3})
      expected = data
      assert MinMaxScaler.fit_transform(data) == expected
    end
  end
end
