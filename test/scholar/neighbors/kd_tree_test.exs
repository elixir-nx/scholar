defmodule Scholar.Neighbors.KDTreeTest do
  use Scholar.Case, async: true
  alias Scholar.Neighbors.KDTree
  doctest KDTree

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
    test "iota" do
      tree = KDTree.fit(Nx.iota({5, 2}))
      assert tree.levels == 3
      assert tree.indices == Nx.u32([3, 1, 4, 0, 2])
      assert tree.num_neighbors == 3
    end

    test "float" do
      tree = KDTree.fit(Nx.as_type(example(), :f32))
      assert tree.levels == 4
      assert Nx.to_flat_list(tree.indices) == [1, 5, 9, 3, 6, 2, 8, 0, 7, 4]
      assert tree.num_neighbors == 3
    end

    test "sample" do
      tree = KDTree.fit(example())
      assert tree.levels == 4
      assert Nx.to_flat_list(tree.indices) == [1, 5, 9, 3, 6, 2, 8, 0, 7, 4]
      assert tree.num_neighbors == 3
    end
  end

  defp x do
    Nx.tensor([
      [3, 6, 7, 5],
      [9, 8, 5, 4],
      [4, 4, 4, 1],
      [9, 4, 5, 6],
      [6, 4, 5, 7],
      [4, 5, 3, 3],
      [4, 5, 7, 8],
      [9, 4, 4, 5],
      [8, 4, 3, 9],
      [2, 8, 4, 4]
    ])
  end

  defp x_pred do
    Nx.tensor([[4, 3, 8, 4], [1, 6, 1, 1], [3, 7, 9, 2], [5, 2, 1, 2]])
  end

  describe "predict knn" do
    test "all defaults" do
      kdtree = KDTree.fit(x())
      {indices, distances} = KDTree.predict(kdtree, x_pred())

      assert indices == Nx.tensor([[0, 6, 4], [5, 2, 9], [0, 9, 2], [5, 2, 7]])

      assert_all_close(
        distances,
        Nx.tensor([
          [3.464101552963257, 4.582575798034668, 4.795831680297852],
          [4.242640495300293, 4.690415859222412, 4.795831680297852],
          [3.7416574954986572, 5.5677642822265625, 6.0],
          [3.872983455657959, 3.872983455657959, 6.164413928985596]
        ])
      )
    end

    test "metric set to {:minkowski, 1.5}" do
      kdtree = KDTree.fit(x(), metric: {:minkowski, 1.5})
      {indices, distances} = KDTree.predict(kdtree, x_pred())

      assert indices == Nx.tensor([[0, 6, 2], [5, 2, 9], [0, 9, 2], [5, 2, 7]])

      assert_all_close(
        distances,
        Nx.tensor([
          [4.065119743347168, 5.191402435302734, 5.862917423248291],
          [5.198591709136963, 5.591182708740234, 5.869683265686035],
          [4.334622859954834, 6.35192346572876, 6.9637274742126465],
          [4.649191856384277, 4.649191856384277, 7.664907932281494]
        ])
      )
    end

    test "k set to 4" do
      kdtree = KDTree.fit(x(), num_neighbors: 4)
      {indices, distances} = KDTree.predict(kdtree, x_pred())

      assert indices == Nx.tensor([[0, 6, 4, 2], [5, 2, 9, 0], [0, 9, 2, 5], [5, 2, 7, 4]])

      assert_all_close(
        distances,
        Nx.tensor([
          [3.464101552963257, 4.582575798034668, 4.795831680297852, 5.099019527435303],
          [4.242640495300293, 4.690415859222412, 4.795831680297852, 7.4833149909973145],
          [3.7416574954986572, 5.5677642822265625, 6.0, 6.480740547180176],
          [3.872983455657959, 3.872983455657959, 6.164413928985596, 6.78233003616333]
        ])
      )
    end

    test "float type data" do
      kdtree = KDTree.fit(x() |> Nx.as_type(:f64), num_neighbors: 4)
      {indices, distances} = KDTree.predict(kdtree, x_pred())

      assert indices == Nx.tensor([[0, 6, 4, 2], [5, 2, 9, 0], [0, 9, 2, 5], [5, 2, 7, 4]])

      assert_all_close(
        distances,
        Nx.tensor([
          [3.464101552963257, 4.582575798034668, 4.795831680297852, 5.099019527435303],
          [4.242640495300293, 4.690415859222412, 4.795831680297852, 7.4833149909973145],
          [3.7416574954986572, 5.5677642822265625, 6.0, 6.480740547180176],
          [3.872983455657959, 3.872983455657959, 6.164413928985596, 6.78233003616333]
        ])
      )
    end
  end
end
