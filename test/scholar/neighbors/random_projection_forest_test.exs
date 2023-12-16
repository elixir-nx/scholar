defmodule Scholar.Neighbors.RandomProjectionForestTest do
  use ExUnit.Case, async: true
  alias Scholar.Neighbors.RandomProjectionForest
  doctest RandomProjectionForest

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

  describe "fit" do
    test "shape" do
      forest = RandomProjectionForest.fit_unbounded(example(), num_trees: 4, min_leaf_size: 3)
      assert forest.depth == 1
      assert forest.leaf_size == 5
      assert forest.num_trees == 4
      assert forest.indices.shape == {4, 10}
      assert forest.data.shape == {10, 2}
      assert forest.hyperplanes.shape == {4, 1, 2}
      assert forest.medians.shape == {4, 1}
    end
  end
end
