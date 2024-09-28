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

      forest =
        RandomProjectionForest.fit(tensor, num_neighbors: 2, num_trees: 4, min_leaf_size: 3)

      assert forest.num_neighbors == 2
      assert forest.depth == 1
      assert forest.leaf_size == 5
      assert forest.num_trees == 4
      assert forest.indices.shape == {4, 10}
      assert forest.data.shape == {10, 2}
      assert forest.hyperplanes.shape == {4, 1, 2}
      assert forest.medians.shape == {4, 1}
    end
  end

  describe "predict" do
    test "shape" do
      tensor = example()

      forest =
        RandomProjectionForest.fit(tensor, num_neighbors: 2, num_trees: 4, min_leaf_size: 3)

      {neighbor_indices, neighbor_distances} =
        RandomProjectionForest.predict(forest, Nx.tensor([[20, 30], [30, 50]]))

      assert Nx.shape(neighbor_indices) == {2, 2}
      assert Nx.shape(neighbor_distances) == {2, 2}
    end

    test "every point is its own neighbor when num_neighbors is 1" do
      key = Nx.Random.key(12)
      {tensor, key} = Nx.Random.uniform(key, shape: {1000, 10})
      size = Nx.axis_size(tensor, 0)

      forest =
        RandomProjectionForest.fit(tensor,
          num_neighbors: 1,
          num_trees: 1,
          min_leaf_size: 1,
          key: key
        )

      {neighbors, distances} = RandomProjectionForest.predict(forest, tensor)
      assert Nx.flatten(neighbors) == Nx.iota({size}, type: :u32)
      assert Nx.flatten(distances) == Nx.broadcast(0.0, {size})
    end

    test "every point is its own neighbor when num_neighbors is 1 and size is power of two" do
      key = Nx.Random.key(12)
      {tensor, key} = Nx.Random.uniform(key, shape: {1024, 10})
      size = Nx.axis_size(tensor, 0)

      forest =
        RandomProjectionForest.fit(tensor,
          num_neighbors: 1,
          num_trees: 1,
          min_leaf_size: 1,
          key: key
        )

      {neighbors, distances} = RandomProjectionForest.predict(forest, tensor)
      assert Nx.flatten(neighbors) == Nx.iota({size}, type: :u32)
      assert Nx.flatten(distances) == Nx.broadcast(0.0, {size})
    end
  end
end
