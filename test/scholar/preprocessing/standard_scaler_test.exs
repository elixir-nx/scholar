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

    test "leaves data as it is when variance is zero" do
      data = 42.0
      expected = Nx.tensor(data)
      assert StandardScaler.fit_transform(data) == expected
    end
  end
end
