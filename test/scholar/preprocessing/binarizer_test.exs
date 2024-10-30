defmodule Scholar.Preprocessing.BinarizerTest do
  use Scholar.Case, async: true
  alias Scholar.Preprocessing.Binarizer
  doctest Binarizer

  describe "binarization" do
    test "binarize with positive threshold" do
      tensor = Nx.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [-2.0, -1.0, 0.0]])

      jit_binarizer = Nx.Defn.jit(&Binarizer.fit_transform/2)

      result = jit_binarizer.(tensor, threshold: 2.0)

      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    end

    test "binarize values with default threshold" do
      tensor = Nx.tensor([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0], [-2.0, 1.0, 0.0]])

      result = Binarizer.fit_transform(tensor)

      assert Nx.to_flat_list(result) == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    end

    test "binarize with threshold less than 0" do
      tensor = Nx.tensor([[0.0, 0.5, -0.5], [-0.1, -0.2, -0.3]])
      jit_binarizer = Nx.Defn.jit(&Binarizer.fit_transform/2)

      result = jit_binarizer.(tensor, threshold: -0.2)

      assert Nx.to_flat_list(result) == [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    end
  end
end
