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
      tensor = example()
      forest = RandomProjectionForest.fit(tensor, num_trees: 4, min_leaf_size: 3)
      assert forest.depth == 1
      assert forest.leaf_size == 5
      assert forest.num_trees == 4
      assert forest.indices.shape == {4, 10}
      assert forest.data.shape == {10, 2}
      assert forest.hyperplanes.shape == {4, 1, 2}
      assert forest.medians.shape == {4, 1}
    end
  end

  defp x do
    key = Nx.Random.key(12)
    Nx.Random.uniform(key, shape: {1024, 10}) |> elem(0)
  end

  describe "predict" do
    test "shape" do
      tensor = example()
      forest = RandomProjectionForest.fit(tensor, num_trees: 4, min_leaf_size: 3)
      leaf_indices = RandomProjectionForest.predict(forest, Nx.tensor([[20, 30], [30, 50]]))
      assert Nx.shape(leaf_indices) == {2, forest.num_trees, forest.leaf_size}
    end

    test "every point is its own leaf when leaf_size is 1" do
      key = Nx.Random.key(12)
      tensor = x()
      forest = RandomProjectionForest.fit(tensor, num_trees: 1, min_leaf_size: 1)
      leaf_indices = RandomProjectionForest.predict(forest, tensor)
      assert Nx.flatten(leaf_indices) == Nx.iota({Nx.axis_size(tensor, 0)}, type: :u32)
    end
  end
end
