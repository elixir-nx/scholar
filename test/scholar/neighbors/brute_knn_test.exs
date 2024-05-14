defmodule Scholar.Neighbors.BruteKNNTest do
  use ExUnit.Case, async: true
  alias Scholar.Neighbors.BruteKNN
  doctest BruteKNN

  defp data do
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

  defp query do
    Nx.tensor([
      [12, 23],
      [55, 30],
      [41, 57],
      [64, 72],
      [26, 39]
    ])
  end

  defp result do
    neighbor_indices =
      Nx.tensor(
        [
          [0, 5, 3],
          [7, 3, 2],
          [6, 1, 9],
          [8, 9, 1],
          [5, 4, 3]
        ],
        type: :u64
      )

    neighbor_distances =
      Nx.tensor([
        [8.246211051940918, 20.2237491607666, 29.73213768005371],
        [14.142135620117188, 15.29705810546875, 15.81138801574707],
        [3.1622776985168457, 7.8102498054504395, 15.620499610900879],
        [3.605551242828369, 12.083045959472656, 20.124610900878906],
        [11.704699516296387, 15.033296585083008, 15.231546401977539]
      ])

    {neighbor_indices, neighbor_distances}
  end

  describe "fit" do
    test "default" do
      data = data()
      k = 3
      model = BruteKNN.fit(data, num_neighbors: k)
      assert model.num_neighbors == 3
      assert model.data == data
      assert model.batch_size == nil
    end

    test "custom metric and batch_size" do
      data = data()
      k = 3
      metric = &Scholar.Metrics.Distance.minkowski/2
      batch_size = 2
      model = BruteKNN.fit(data, num_neighbors: k, metric: metric, batch_size: batch_size)
      assert model.num_neighbors == k
      assert model.metric == metric
      assert model.data == data
      assert model.batch_size == batch_size
    end
  end

  describe "predict" do
    test "batch_size = 1" do
      query = query()
      k = 3
      model = BruteKNN.fit(data(), num_neighbors: k, batch_size: 1)
      {neighbors_true, distances_true} = result()
      {neighbors_pred, distances_pred} = BruteKNN.predict(model, query)
      assert neighbors_pred == neighbors_true
      assert distances_pred == distances_true
    end

    test "batch_size = 2" do
      query = query()
      k = 3
      model = BruteKNN.fit(data(), num_neighbors: k, batch_size: 2)
      {neighbors_true, distances_true} = result()
      {neighbors_pred, distances_pred} = BruteKNN.predict(model, query)
      assert neighbors_pred == neighbors_true
      assert distances_pred == distances_true
    end

    test "batch_size = 5" do
      query = query()
      k = 3
      model = BruteKNN.fit(data(), num_neighbors: k, batch_size: 5)
      {neighbors_true, distances_true} = result()
      {neighbors_pred, distances_pred} = BruteKNN.predict(model, query)
      assert neighbors_pred == neighbors_true
      assert distances_pred == distances_true
    end

    test "batch_size = 10" do
      query = query()
      k = 3
      model = BruteKNN.fit(data(), num_neighbors: k, batch_size: 10)
      {neighbors_true, distances_true} = result()
      {neighbors_pred, distances_pred} = BruteKNN.predict(model, query)

      assert neighbors_pred ==
               neighbors_true

      assert distances_pred == distances_true
    end

    test "custom metric" do
      model = BruteKNN.fit(data(), num_neighbors: 3, batch_size: 1, metric: :cosine)
      assert {_, _} = BruteKNN.predict(model, query())
    end
  end
end
