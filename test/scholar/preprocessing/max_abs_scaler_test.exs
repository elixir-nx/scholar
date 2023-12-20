defmodule Scholar.Preprocessing.MaxAbsScalerTest do
  use Scholar.Case, async: true
  alias Scholar.Preprocessing.MaxAbsScaler

  doctest MaxAbsScaler

  describe "fit_transform/2" do
    test "set axes to [0]" do
      data = Nx.tensor([[1, -1, 2], [3, 0, 0], [0, 1, -1], [2, 3, 1]])

      expected =
        Nx.tensor([
          [0.3333333432674408, -0.3333333432674408, 1.0],
          [1.0, 0.0, 0.0],
          [0.0, 0.3333333432674408, -0.5],
          [0.6666666865348816, 1.0, 0.5]
        ])

      assert_all_close(MaxAbsScaler.fit_transform(data, axes: [0]), expected)
    end

    test "Work in case where tensor contains only zeros" do
      data = Nx.broadcast(Nx.f32(0), {3, 3})
      expected = data
      assert MaxAbsScaler.fit_transform(data) == expected
    end
  end
end
