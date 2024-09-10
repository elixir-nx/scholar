defmodule Scholar.Neighbors.RadiusNNRegressorTest do
  use Scholar.Case, async: true
  alias Scholar.Neighbors.RadiusNNRegressor
  doctest RadiusNNRegressor

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

  defp y do
    Nx.tensor([0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
  end

  defp x_pred do
    Nx.tensor([[4, 3, 8, 4], [1, 6, 1, 1], [3, 7, 9, 2], [5, 2, 1, 2]])
  end

  describe "predict" do
    test "predict with weights set to :distance" do
      model =
        RadiusNNRegressor.fit(x(), y(),
          num_classes: 2,
          radius: 10,
          weights: :distance
        )

      predictions = RadiusNNRegressor.predict(model, x_pred())
      assert_all_close(predictions, Nx.tensor([0.69033845, 0.71773642, 0.68217609, 0.75918273]))
    end

    test "predict with weights set to :distance and with specific metric" do
      model =
        RadiusNNRegressor.fit(x(), y(),
          num_classes: 2,
          radius: 10,
          weights: :distance,
          metric: :cosine
        )

      predictions = RadiusNNRegressor.predict(model, x_pred())
      assert_all_close(predictions, Nx.tensor([0.683947, 0.54694187, 0.59806132, 0.86398641]))
    end

    test "predict with weights set to :distance and with specific metric and 2d labels" do
      y =
        Nx.tensor([[1, 4], [0, 3], [2, 5], [0, 3], [0, 3], [1, 4], [2, 5], [0, 3], [1, 4], [2, 5]])

      model =
        RadiusNNRegressor.fit(x(), y,
          num_classes: 3,
          radius: 10,
          weights: :distance,
          metric: :cosine
        )

      predictions = RadiusNNRegressor.predict(model, x_pred())

      assert_all_close(
        predictions,
        Nx.tensor([
          [0.99475077, 3.99475077],
          [1.20828527, 4.20828527],
          [1.15227075, 4.15227075],
          [0.37743229, 3.37743229]
        ])
      )
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
                     RadiusNNRegressor.fit(x, y, num_classes: 5)
                   end
    end

    test "wrong shape of y" do
      x = Nx.tensor([[1], [2], [3], [4], [5], [6]])
      y = Nx.tensor([[[1, 2, 3, 4, 5]]])

      assert_raise ArgumentError,
                   "expected labels to have shape {num_samples} or {num_samples, num_outputs},
            got tensor with shape: {1, 1, 5}",
                   fn ->
                     RadiusNNRegressor.fit(x, y, num_classes: 5)
                   end
    end

    test "incompatible shapes of x and y" do
      x = Nx.tensor([[1], [2], [3], [4], [5], [6]])
      y = Nx.tensor([1, 2, 3, 4, 5])

      assert_raise ArgumentError,
                   "expected labels to have the same size of the first axis as data,
      got: 6 != 5",
                   fn ->
                     RadiusNNRegressor.fit(x, y, num_classes: 5)
                   end
    end
  end
end
