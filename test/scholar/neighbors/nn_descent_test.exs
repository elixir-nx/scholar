defmodule Scholar.Neighbors.NNDescentTest do
  use ExUnit.Case, async: true
  alias Scholar.Neighbors.NNDescent
  doctest NNDescent

  test "every point is its own neighbor when num_neighbors is 1" do
    key = Nx.Random.key(12)
    {tensor, key} = Nx.Random.uniform(key, shape: {10, 5})
    size = Nx.axis_size(tensor, 0)

    %NNDescent{nearest_neighbors: nearest_neighbors, distances: distances} =
      NNDescent.fit(tensor,
        num_neighbors: 1,
        key: key
      )

    assert Nx.flatten(nearest_neighbors) == Nx.iota({size}, type: :s64)
    assert Nx.flatten(distances) == Nx.broadcast(0.0, {size})
  end
end
