defmodule Scholar.Neighbors.KNearestNeighborsTest do
  use Scholar.Case, async: true
  alias Scholar.Neighbors.KNearestNeighbors
  doctest KNearestNeighbors

  @x Nx.tensor([
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

  @y Nx.tensor([0, 1, 1, 1, 1, 1, 1, 1, 0, 0])

  @x_pred Nx.tensor([[4, 3, 8, 4], [1, 6, 1, 1], [3, 7, 9, 2], [5, 2, 1, 2]])

  describe "fit" do
    test "fit with default parameters - :num_classes set to 2" do
      model = KNearestNeighbors.fit(@x, @y, num_classes: 2)

      assert model.default_num_neighbors == 5
      assert model.weights == :uniform
      assert model.task == :classification
      assert model.metric == {:minkowski, 2}
      assert model.num_classes == 2
      assert model.data == @x
      assert model.labels == @y
    end
  end

  describe "predict" do
    test "predict with default values - classification task" do
      model = KNearestNeighbors.fit(@x, @y, num_classes: 2, num_neighbors: 3)
      predictions = KNearestNeighbors.predict(model, @x_pred)
      assert predictions == Nx.tensor([1, 1, 0, 1])
    end

    test "predict with default values - regression task" do
      model = KNearestNeighbors.fit(@x, @y, num_classes: 2, num_neighbors: 3, task: :regression)
      predictions = KNearestNeighbors.predict(model, @x_pred)
      assert_all_close(predictions, Nx.tensor([0.66666667, 0.66666667, 0.33333333, 1.0]))
    end

    test "predict with weights set to :distance - classification task" do
      model = KNearestNeighbors.fit(@x, @y, num_classes: 2, num_neighbors: 3, weights: :distance)
      predictions = KNearestNeighbors.predict(model, @x_pred)
      assert predictions == Nx.tensor([1, 1, 0, 1])
    end

    test "predict with weights set to :distance - regression task" do
      model =
        KNearestNeighbors.fit(@x, @y,
          num_classes: 2,
          num_neighbors: 3,
          task: :regression,
          weights: :distance
        )

      predictions = KNearestNeighbors.predict(model, @x_pred)
      assert_all_close(predictions, Nx.tensor([0.59648849, 0.68282796, 0.2716506, 1.0]))
    end

    test "predict with weights set to :distance and with specific metric - classification task" do
      model =
        KNearestNeighbors.fit(@x, @y,
          num_classes: 2,
          num_neighbors: 3,
          weights: :distance,
          metric: {:minkowski, 1.5}
        )

      predictions = KNearestNeighbors.predict(model, @x_pred)
      assert predictions == Nx.tensor([1, 1, 0, 1])
    end

    test "predict with weights set to :distance and with specific metric - regression task" do
      model =
        KNearestNeighbors.fit(@x, @y,
          num_classes: 2,
          num_neighbors: 3,
          task: :regression,
          weights: :distance,
          metric: :cosine
        )

      predictions = KNearestNeighbors.predict(model, @x_pred)
      assert_all_close(predictions, Nx.tensor([0.5736568, 0.427104, 0.33561941, 1.0]))
    end

    test "predict with weights set to :distance and with specific metric and 2d labels - regression task" do
      y =
        Nx.tensor([[1, 4], [0, 3], [2, 5], [0, 3], [0, 3], [1, 4], [2, 5], [0, 3], [1, 4], [2, 5]])

      model =
        KNearestNeighbors.fit(@x, y,
          num_classes: 3,
          num_neighbors: 3,
          task: :regression,
          weights: :distance,
          metric: :cosine
        )

      predictions = KNearestNeighbors.predict(model, @x_pred)

      assert_all_close(
        predictions,
        Nx.tensor([
          [1.11344606, 4.11344606],
          [1.3915288, 4.3915288],
          [1.53710155, 4.53710155],
          [0.0, 3.0]
        ])
      )
    end

    test "predict with weights set to :distance and with x_pred that contains sample with zero-distance" do
      x_pred = Nx.tensor([[3, 6, 7, 5], [1, 6, 1, 1], [3, 7, 9, 2], [5, 2, 1, 2]])
      model = KNearestNeighbors.fit(@x, @y, num_classes: 2, num_neighbors: 3, weights: :distance)
      predictions = KNearestNeighbors.predict(model, x_pred)
      assert predictions == Nx.tensor([0, 1, 0, 1])
    end
  end

  describe "predict_proba" do
    test "predict_proba with default values" do
      model = KNearestNeighbors.fit(@x, @y, num_classes: 2, num_neighbors: 3)
      predictions = KNearestNeighbors.predict_proba(model, @x_pred)

      assert_all_close(
        predictions,
        Nx.tensor([
          [0.33333333, 0.66666667],
          [0.33333333, 0.66666667],
          [0.66666667, 0.33333333],
          [0.0, 1.0]
        ])
      )
    end

    test "predict_proba with weights set to :distance" do
      model = KNearestNeighbors.fit(@x, @y, num_classes: 2, num_neighbors: 3, weights: :distance)
      predictions = KNearestNeighbors.predict_proba(model, @x_pred)

      assert_all_close(
        predictions,
        Nx.tensor([
          [0.40351151, 0.59648849],
          [0.31717204, 0.68282796],
          [0.7283494, 0.2716506],
          [0.0, 1.0]
        ])
      )
    end

    test "predict_proba with weights set to :distance and with specific metric" do
      model =
        KNearestNeighbors.fit(@x, @y,
          num_classes: 2,
          num_neighbors: 3,
          weights: :distance,
          metric: {:minkowski, 1.5}
        )

      predictions = KNearestNeighbors.predict_proba(model, @x_pred)

      assert_all_close(
        predictions,
        Nx.tensor([
          [0.40381038, 0.59618962],
          [0.31457406, 0.68542594],
          [0.72993802, 0.27006198],
          [0.0, 1.0]
        ])
      )
    end

    test "predict_proba with weights set to :distance and with x_pred that contains sample with zero-distance" do
      x_pred = Nx.tensor([[3, 6, 7, 5], [1, 6, 1, 1], [3, 7, 9, 2], [5, 2, 1, 2]])
      model = KNearestNeighbors.fit(@x, @y, num_classes: 2, num_neighbors: 3, weights: :distance)
      predictions = KNearestNeighbors.predict_proba(model, x_pred)

      assert_all_close(
        predictions,
        Nx.tensor([
          [1.0, 0.0],
          [0.31717204, 0.68282796],
          [0.7283494, 0.2716506],
          [0.0, 1.0]
        ])
      )
    end
  end

  describe "k_neighbors" do
    test "k_neighbors with default values" do
      model = KNearestNeighbors.fit(@x, @y, num_classes: 2, num_neighbors: 3)
      {distances, indices} = KNearestNeighbors.k_neighbors(model, @x_pred)

      assert_all_close(
        distances,
        Nx.tensor([
          [3.46410162, 4.58257569, 4.79583152],
          [4.24264069, 4.69041576, 4.79583152],
          [3.74165739, 5.56776436, 6.0],
          [3.87298335, 3.87298335, 6.164414]
        ])
      )

      assert indices == Nx.tensor([[0, 6, 4], [5, 2, 9], [0, 9, 2], [2, 5, 7]])
    end

    test "k_neighbors with specific metric" do
      model =
        KNearestNeighbors.fit(@x, @y, num_classes: 2, num_neighbors: 3, metric: {:minkowski, 1.5})

      {distances, indices} = KNearestNeighbors.k_neighbors(model, @x_pred)

      assert_all_close(
        distances,
        Nx.tensor([
          [4.06511982, 5.19140207, 5.86291731],
          [5.19859142, 5.59118249, 5.86968282],
          [4.33462287, 6.35192346, 6.96372712],
          [4.6491916, 4.6491916, 7.6649073]
        ])
      )

      assert indices == Nx.tensor([[0, 6, 2], [5, 2, 9], [0, 9, 2], [2, 5, 7]])
    end
  end

  describe "errors" do
    test "wrong shape of x" do
      x = Nx.tensor([1, 2, 3, 4, 5])
      y = Nx.tensor([1, 2, 3, 4, 5])

      assert_raise ArgumentError,
                   "expected input tensor to have shape {n_samples, n_features} or {num_samples, num_samples},
             got tensor with shape: {5}",
                   fn ->
                     KNearestNeighbors.fit(x, y, num_classes: 5)
                   end
    end

    test "wrong shape of y" do
      x = Nx.tensor([[1], [2], [3], [4], [5], [6]])
      y = Nx.tensor([[[1, 2, 3, 4, 5]]])

      assert_raise ArgumentError,
                   "expected labels to have shape {num_samples} or {num_samples, num_outputs},
            got tensor with shape: {1, 1, 5}",
                   fn ->
                     KNearestNeighbors.fit(x, y, num_classes: 5)
                   end
    end

    test "incompatible shapes of x and y" do
      x = Nx.tensor([[1], [2], [3], [4], [5], [6]])
      y = Nx.tensor([1, 2, 3, 4, 5])

      assert_raise ArgumentError,
                   "expected labels to have the same size of the first axis as data,
      got: 6 != 5",
                   fn ->
                     KNearestNeighbors.fit(x, y, num_classes: 5)
                   end
    end
  end
end
