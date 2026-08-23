defmodule Scholar.Preprocessing.StandardScalerTest do
  use Scholar.Case, async: true
  alias Scholar.Preprocessing.StandardScaler

  doctest StandardScaler

  describe "fit_transform/2" do
    test "applies standard scaling to data" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])

      expected =
        Nx.tensor([
          [0.5212860703468323, -1.3553436994552612, 1.4596009254455566],
          [1.4596009254455566, -0.4170288145542145, -0.4170288145542145],
          [-0.4170288145542145, 0.5212860703468323, -1.3553436994552612]
        ])

      assert_all_close(StandardScaler.fit_transform(data), expected)
    end

    test "centers data around zero when variance is zero" do
      data = 42.0
      expected = Nx.tensor(0.0)
      assert StandardScaler.fit_transform(data) == expected
    end

    test "centers a constant feature to zero while leaving other features scaled normally" do
      data = Nx.tensor([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])

      expected =
        Nx.tensor([
          [-1.2247448, 0.0],
          [0.0, 0.0],
          [1.2247448, 0.0]
        ])

      assert_all_close(StandardScaler.fit_transform(data, axes: [0]), expected)

      scaler = StandardScaler.fit(data, axes: [0])
      assert_all_close(scaler.mean, Nx.tensor([[2.0, 5.0]]))
    end
  end
end
