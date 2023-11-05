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

  describe "unbanded" do
    test "sample" do
      assert %Scholar.Neighbors.KDTree{levels: 4, indexes: indexes} =
               Scholar.Neighbors.KDTree.unbanded(example(), compiler: EXLA.Defn)

      assert Nx.to_flat_list(indexes) == [1, 5, 9, 3, 6, 2, 8, 0, 7, 4]
    end

    test "corner cases" do
      assert Scholar.Neighbors.KDTree.unbanded(Nx.iota({1, 2}), compiler: EXLA.Defn) ==
               %Scholar.Neighbors.KDTree{levels: 1, indexes: Nx.u32([0])}

      assert Scholar.Neighbors.KDTree.unbanded(Nx.iota({2, 2}), compiler: EXLA.Defn) ==
               %Scholar.Neighbors.KDTree{levels: 2, indexes: Nx.u32([1, 0])}
    end
  end

  describe "banded" do
    test "iota" do
      assert Scholar.Neighbors.KDTree.banded(Nx.iota({5, 2}), 10) ==
               %Scholar.Neighbors.KDTree{levels: 3, indexes: Nx.u32([3, 1, 4, 0, 2])}
    end

    test "sample" do
      input = Nx.u32([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

      assert Nx.Defn.jit_apply(
               &Scholar.Neighbors.KDTree.banded_segment_begin(&1, 4, 10),
               [input]
             ) == Nx.u32([0, 1, 7, 3, 6, 8, 9, 7, 8, 9])

      assert %Scholar.Neighbors.KDTree{levels: 4, indexes: indexes} =
               Scholar.Neighbors.KDTree.banded(example(), 100)

      assert Nx.to_flat_list(indexes) == [1, 5, 9, 3, 6, 2, 8, 0, 7, 4]
    end
  end
end
