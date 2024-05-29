defmodule Scholar.Neighbors.KNNRegressorTest do
  use Scholar.Case, async: true
  alias Scholar.Neighbors.KNNRegressor
  doctest KNNRegressor

  defp x_train do
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

  defp y_train do
    Nx.tensor([[0], [1], [1], [1], [1], [1], [1], [1], [0], [0]])
  end

  defp x do
    Nx.tensor([[4, 3, 8, 4], [1, 6, 1, 1], [3, 7, 9, 2], [5, 2, 1, 2]])
  end

  describe "fit" do
    test "fit with default parameters" do
      model = KNNRegressor.fit(x_train(), y_train(), num_neighbors: 3)

      assert model.algorithm == Scholar.Neighbors.BruteKNN.fit(x_train(), num_neighbors: 3)
      assert model.labels == y_train()
      assert model.weights == :uniform
    end

    test "fit with k-d tree" do
      model = KNNRegressor.fit(x_train(), y_train(), algorithm: :kd_tree, num_neighbors: 3)

      assert model.algorithm == Scholar.Neighbors.KDTree.fit(x_train(), num_neighbors: 3)
      assert model.labels == y_train()
      assert model.weights == :uniform
    end

    test "fit with random projection forest" do
      key = Nx.Random.key(12)

      model =
        KNNRegressor.fit(x_train(), y_train(),
          algorithm: :random_projection_forest,
          num_neighbors: 3,
          num_trees: 4,
          key: key
        )

      assert model.algorithm ==
               Scholar.Neighbors.RandomProjectionForest.fit(x_train(),
                 num_neighbors: 3,
                 num_trees: 4,
                 key: key
               )

      assert model.labels == y_train()
      assert model.weights == :uniform
    end
  end

  describe "predict" do
    test "predict with default parameters" do
      model = KNNRegressor.fit(x_train(), y_train(), num_neighbors: 3)
      y_pred = KNNRegressor.predict(model, x())
      assert_all_close(y_pred, Nx.tensor([[0.66666667], [0.66666667], [0.33333333], [1.0]]))
    end

    test "predict with weights set to :distance" do
      model = KNNRegressor.fit(x_train(), y_train(), num_neighbors: 3, weights: :distance)
      y_pred = KNNRegressor.predict(model, x())
      assert_all_close(y_pred, Nx.tensor([[0.59648849], [0.68282796], [0.2716506], [1.0]]))
    end

    test "predict with cosine metric and weights set to :distance" do
      model =
        KNNRegressor.fit(x_train(), y_train(),
          num_neighbors: 3,
          metric: :cosine,
          weights: :distance
        )

      y_pred = KNNRegressor.predict(model, x())
      assert_all_close(y_pred, Nx.tensor([[0.5736568], [0.427104], [0.33561941], [1.0]]))
    end

    test "predict with 2D labels" do
      y =
        Nx.tensor([[1, 4], [0, 3], [2, 5], [0, 3], [0, 3], [1, 4], [2, 5], [0, 3], [1, 4], [2, 5]])

      model = KNNRegressor.fit(x_train(), y, num_neighbors: 3)
      y_pred = KNNRegressor.predict(model, x())

      assert_all_close(
        y_pred,
        Nx.tensor([
          [1.0, 4.0],
          [1.6666666269302368, 4.666666507720947],
          [1.6666666269302368, 4.666666507720947],
          [1.0, 4.0]
        ])
      )
    end

    test "predict with 2D labels, cosine metric and weights set to :distance" do
      y =
        Nx.tensor([[1, 4], [0, 3], [2, 5], [0, 3], [0, 3], [1, 4], [2, 5], [0, 3], [1, 4], [2, 5]])

      model =
        KNNRegressor.fit(x_train(), y, num_neighbors: 3, metric: :cosine, weights: :distance)

      y_pred = KNNRegressor.predict(model, x())

      assert_all_close(
        y_pred,
        Nx.tensor([
          [1.11344606, 4.11344606],
          [1.3915288, 4.3915288],
          [1.53710155, 4.53710155],
          [0.0, 3.0]
        ])
      )
    end
  end
end
