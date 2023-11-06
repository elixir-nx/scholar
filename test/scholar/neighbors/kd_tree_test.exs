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

  describe "unbounded" do
    test "sample" do
      assert %Scholar.Neighbors.KDTree{levels: 4, indexes: indexes} =
               Scholar.Neighbors.KDTree.unbounded(example(), compiler: EXLA)

      assert Nx.to_flat_list(indexes) == [1, 5, 9, 3, 6, 2, 8, 0, 7, 4]
    end

    test "float" do
      assert %Scholar.Neighbors.KDTree{levels: 4, indexes: indexes} =
               Scholar.Neighbors.KDTree.unbounded(example() |> Nx.as_type(:f32),
                 compiler: EXLA
               )

      assert Nx.to_flat_list(indexes) == [1, 5, 9, 3, 6, 2, 8, 0, 7, 4]
    end

    test "corner cases" do
      assert %Scholar.Neighbors.KDTree{levels: 1, indexes: indexes} =
               Scholar.Neighbors.KDTree.unbounded(Nx.iota({1, 2}), compiler: EXLA)

      assert indexes == Nx.u32([0])

      assert %Scholar.Neighbors.KDTree{levels: 2, indexes: indexes} =
               Scholar.Neighbors.KDTree.unbounded(Nx.iota({2, 2}), compiler: EXLA)

      assert indexes == Nx.u32([1, 0])
    end
  end

  describe "bounded" do
    test "iota" do
      assert %Scholar.Neighbors.KDTree{levels: 3, indexes: indexes} =
               Scholar.Neighbors.KDTree.bounded(Nx.iota({5, 2}), 10)

      assert indexes == Nx.u32([3, 1, 4, 0, 2])
    end

    test "float" do
      assert %Scholar.Neighbors.KDTree{levels: 4, indexes: indexes} =
               Scholar.Neighbors.KDTree.bounded(example() |> Nx.as_type(:f32), 100)

      assert Nx.to_flat_list(indexes) == [1, 5, 9, 3, 6, 2, 8, 0, 7, 4]
    end

    test "sample" do
      assert %Scholar.Neighbors.KDTree{levels: 4, indexes: indexes} =
               Scholar.Neighbors.KDTree.bounded(example(), 100)

      assert Nx.to_flat_list(indexes) == [1, 5, 9, 3, 6, 2, 8, 0, 7, 4]
    end
  end
end
