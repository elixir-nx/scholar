defmodule Scholar.Neighbors.KDTreeTest do
  use ExUnit.Case, async: true
  doctest Scholar.Neighbors.KDTree

  defp example do
    Nx.tensor([
      [10, 15],
      [46, 63],
      [68, 21],
      [40, 33],
      [25, 54],
      [15, 43],
      [44, 58],
      [45, 40],
      [62, 69],
      [53, 67]
    ])
  end

  test "unbanded" do
    assert %Scholar.Neighbors.KDTree{levels: 4, indexes: indexes} =
             Scholar.Neighbors.KDTree.unbanded(example(), compiler: EXLA.Defn)

    assert Nx.to_flat_list(indexes) == [1, 5, 9, 3, 6, 2, 8, 0, 7, 4]
  end
end
