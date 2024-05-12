defmodule Scholar.Neighbors.KNNClassifierTest do
  use Scholar.Case, async: true
  alias Scholar.Neighbors.KNNClassifier
  doctest KNNClassifier

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
    Nx.tensor([0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
  end

  defp x do
    Nx.tensor([[4, 3, 8, 4], [1, 6, 1, 1], [3, 7, 9, 2], [5, 2, 1, 2]])
  end

  describe "fit" do
    test "fit with default parameters - :num_classes set to 2" do
      model = KNNClassifier.fit(x_train(), y_train(), num_neighbors: 3, num_classes: 2)

      assert model.algorithm == Scholar.Neighbors.BruteKNN.fit(x_train(), num_neighbors: 3)
      assert model.num_classes == 2
      assert model.labels == y_train()
      assert model.weights == :uniform
    end

    test "fit with k-d tree" do
      model =
        KNNClassifier.fit(x_train(), y_train(),
          algorithm: :kd_tree,
          num_neighbors: 3,
          num_classes: 2
        )

      assert model.algorithm == Scholar.Neighbors.KDTree.fit(x_train(), num_neighbors: 3)
      assert model.num_classes == 2
      assert model.labels == y_train()
      assert model.weights == :uniform
    end
  end

  describe "predict" do
    test "predict with default values" do
      model = KNNClassifier.fit(x_train(), y_train(), num_neighbors: 3, num_classes: 2)
      labels_pred = KNNClassifier.predict(model, x())
      assert labels_pred == Nx.tensor([1, 1, 0, 1])
    end

    test "predict with k-d tree" do
      model =
        KNNClassifier.fit(x_train(), y_train(),
          algorithm: :kd_tree,
          num_neighbors: 3,
          num_classes: 2
        )

      labels_pred = KNNClassifier.predict(model, x())
      assert labels_pred == Nx.tensor([1, 1, 0, 1])
    end

    test "predict with weights set to :distance" do
      model =
        KNNClassifier.fit(x_train(), y_train(),
          num_neighbors: 3,
          num_classes: 2,
          weights: :distance
        )

      labels_pred = KNNClassifier.predict(model, x())
      assert labels_pred == Nx.tensor([1, 1, 0, 1])
    end

    test "predict with specific metric and weights set to :distance" do
      model =
        KNNClassifier.fit(x_train(), y_train(),
          num_neighbors: 3,
          num_classes: 2,
          metric: {:minkowski, 1.5},
          weights: :distance
        )

      labels_pred = KNNClassifier.predict(model, x())
      assert labels_pred == Nx.tensor([1, 1, 0, 1])
    end

    test "predict with weights set to :distance and with x that contains sample with zero-distance" do
      x = Nx.tensor([[3, 6, 7, 5], [1, 6, 1, 1], [3, 7, 9, 2], [5, 2, 1, 2]])

      model =
        KNNClassifier.fit(x_train(), y_train(),
          num_neighbors: 3,
          num_classes: 2,
          weights: :distance
        )

      labels_pred = KNNClassifier.predict(model, x)
      assert labels_pred == Nx.tensor([0, 1, 0, 1])
    end
  end
end
