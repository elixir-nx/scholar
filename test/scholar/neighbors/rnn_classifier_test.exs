defmodule Scholar.Neighbors.RadiusNNClassifierTest do
  use Scholar.Case, async: true
  alias Scholar.Neighbors.RadiusNNClassifier
  doctest RadiusNNClassifier

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

  describe "fit" do
    test "fit with default parameters - :num_classes set to 2" do
      model = RadiusNNClassifier.fit(x(), y(), num_classes: 2)

      assert model.weights == :uniform
      assert model.num_classes == 2
      assert model.data == x()
      assert model.labels == y()
    end
  end

  describe "predict" do
    test "predict with default values" do
      model = RadiusNNClassifier.fit(x(), y(), num_classes: 2)
      predictions = RadiusNNClassifier.predict(model, x_pred())
      assert predictions == Nx.tensor([-1, -1, -1, -1])
    end

    test "predict with radius set to 10" do
      model = RadiusNNClassifier.fit(x(), y(), num_classes: 2, radius: 10)
      predictions = RadiusNNClassifier.predict(model, x_pred())
      assert predictions == Nx.tensor([1, 1, 1, 1])
    end

    test "predict with weights set to :distance - classification task" do
      model = RadiusNNClassifier.fit(x(), y(), num_classes: 2, radius: 10, weights: :distance)

      predictions = RadiusNNClassifier.predict(model, x_pred())
      assert predictions == Nx.tensor([1, 1, 1, 1])
    end

    test "predict with weights set to :distance and with specific metric" do
      model =
        RadiusNNClassifier.fit(x(), y(),
          num_classes: 2,
          radius: 10,
          weights: :distance,
          metric: {:minkowski, 1.5}
        )

      predictions = RadiusNNClassifier.predict(model, x_pred())
      assert predictions == Nx.tensor([1, 1, 1, 1])
    end

    test "predict with weights set to :distance and with x_pred that contains sample with zero-distance" do
      x_pred = Nx.tensor([[3, 6, 7, 5], [1, 6, 1, 1], [3, 7, 9, 2], [5, 2, 1, 2]])

      model = RadiusNNClassifier.fit(x(), y(), num_classes: 2, radius: 10, weights: :distance)

      predictions = RadiusNNClassifier.predict(model, x_pred)
      assert predictions == Nx.tensor([0, 1, 1, 1])
    end
  end

  describe "predict_proba" do
    test "predict_proba with default values except radius set to 10" do
      model = RadiusNNClassifier.fit(x(), y(), num_classes: 2, radius: 10)
      {predictions, outliers_mask} = RadiusNNClassifier.predict_probability(model, x_pred())

      assert_all_close(
        predictions,
        Nx.tensor([[0.3, 0.7], [0.25, 0.75], [0.22222222, 0.77777778], [0.3, 0.7]])
      )

      assert_all_close(outliers_mask, Nx.tensor([0, 0, 0, 0], type: :u8))
    end

    test "predict_proba with weights set to :distance" do
      model = RadiusNNClassifier.fit(x(), y(), num_classes: 2, radius: 10, weights: :distance)

      {predictions, outliers_mask} = RadiusNNClassifier.predict_probability(model, x_pred())

      assert_all_close(
        predictions,
        Nx.tensor([
          [0.30966155, 0.69033845],
          [0.28226358, 0.71773642],
          [0.31782391, 0.68217609],
          [0.24081727, 0.75918273]
        ])
      )

      assert_all_close(outliers_mask, Nx.tensor([0, 0, 0, 0], type: :u8))
    end

    test "predict_proba with weights set to :distance and with specific metric" do
      model =
        RadiusNNClassifier.fit(x(), y(),
          num_classes: 2,
          radius: 10,
          weights: :distance,
          metric: {:minkowski, 1.5}
        )

      {predictions, outliers_mask} = RadiusNNClassifier.predict_probability(model, x_pred())

      assert_all_close(
        predictions,
        Nx.tensor([
          [0.30635736, 0.69364264],
          [0.43492793, 0.56507207],
          [0.38180641, 0.61819359],
          [0.28398755, 0.71601245]
        ])
      )

      assert_all_close(outliers_mask, Nx.tensor([0, 0, 0, 0], type: :u8))
    end

    test "predict_proba with weights set to :distance and with x_pred that contains sample with zero-distance" do
      x_pred = Nx.tensor([[3, 6, 7, 5], [1, 6, 1, 1], [3, 7, 9, 2], [5, 2, 1, 2]])

      model = RadiusNNClassifier.fit(x(), y(), num_classes: 2, radius: 10, weights: :distance)

      {predictions, outliers_mask} = RadiusNNClassifier.predict_probability(model, x_pred)

      assert_all_close(
        predictions,
        Nx.tensor([
          [1.0, 0.0],
          [0.28226358, 0.71773642],
          [0.31782391, 0.68217609],
          [0.24081727, 0.75918273]
        ])
      )

      assert_all_close(outliers_mask, Nx.tensor([0, 0, 0, 0], type: :u8))
    end
  end

  describe "radius_neighbors" do
    test "radius_neighbors with default values except radius set to 10" do
      model = RadiusNNClassifier.fit(x(), y(), num_classes: 2, radius: 10)
      {distances, indices} = RadiusNNClassifier.radius_neighbors(model, x_pred())

      assert_all_close(
        distances,
        Nx.tensor([
          [
            3.464101552963257,
            7.681145668029785,
            5.099019527435303,
            6.244997978210449,
            4.795831680297852,
            5.4772257804870605,
            4.582575798034668,
            6.557438373565674,
            8.185352325439453,
            6.7082037925720215
          ],
          [
            7.4833149909973145,
            9.643651008605957,
            4.690415859222412,
            10.440306663513184,
            9.0,
            4.242640495300293,
            9.746794700622559,
            9.643651008605957,
            11.0,
            4.795831680297852
          ],
          [
            3.7416574954986572,
            7.549834251403809,
            6.0,
            8.774964332580566,
            7.681145668029785,
            6.480740547180176,
            6.7082037925720215,
            8.88819408416748,
            10.908712387084961,
            5.5677642822265625
          ],
          [
            8.062257766723633,
            8.485280990600586,
            3.872983455657959,
            7.211102485656738,
            6.78233003616333,
            3.872983455657959,
            9.05538558959961,
            6.164413928985596,
            8.124038696289062,
            7.6157732009887695
          ]
        ])
      )

      assert indices ==
               Nx.tensor(
                 [
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                 ],
                 type: :u8
               )
    end

    test "radius_neighbors with specific metric" do
      model =
        RadiusNNClassifier.fit(x(), y(),
          num_classes: 2,
          radius: 10,
          metric: {:minkowski, 1.5}
        )

      {distances, indices} = RadiusNNClassifier.radius_neighbors(model, x_pred())

      assert_all_close(
        distances,
        Nx.tensor([
          [
            4.065119743347168,
            9.123319625854492,
            5.862917423248291,
            7.418306827545166,
            5.869683265686035,
            6.084571838378906,
            5.191402435302734,
            7.655178070068359,
            9.944668769836426,
            7.85351037979126
          ],
          [
            8.669246673583984,
            11.431800842285156,
            5.591182708740234,
            12.583209037780762,
            11.044746398925781,
            5.198591709136963,
            11.581432342529297,
            11.431800842285156,
            12.987492561340332,
            5.869683265686035
          ],
          [
            4.334622859954834,
            8.894214630126953,
            6.9637274742126465,
            10.881128311157227,
            9.562984466552734,
            7.251026153564453,
            7.696915149688721,
            10.957086563110352,
            13.498462677001953,
            6.35192346572876
          ],
          [
            9.80908489227295,
            10.397183418273926,
            4.649191856384277,
            8.961833000183105,
            8.089634895324707,
            4.649191856384277,
            10.819792747497559,
            7.664907932281494,
            9.519953727722168,
            9.202789306640625
          ]
        ])
      )

      assert indices ==
               Nx.tensor(
                 [
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                   [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                   [1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
                 ],
                 type: :u8
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
                     RadiusNNClassifier.fit(x, y, num_classes: 5)
                   end
    end

    test "wrong shape of y" do
      x = Nx.tensor([[1], [2], [3], [4], [5], [6]])
      y = Nx.tensor([[[1, 2, 3, 4, 5]]])

      assert_raise ArgumentError,
                   "expected labels to have shape {num_samples} or {num_samples, num_outputs},
            got tensor with shape: {1, 1, 5}",
                   fn ->
                     RadiusNNClassifier.fit(x, y, num_classes: 5)
                   end
    end

    test "incompatible shapes of x and y" do
      x = Nx.tensor([[1], [2], [3], [4], [5], [6]])
      y = Nx.tensor([1, 2, 3, 4, 5])

      assert_raise ArgumentError,
                   "expected labels to have the same size of the first axis as data,
      got: 6 != 5",
                   fn ->
                     RadiusNNClassifier.fit(x, y, num_classes: 5)
                   end
    end

    test ":num_classes not provided" do
      x = Nx.tensor([[1], [2], [3], [4], [5], [6]])
      y = Nx.tensor([1, 2, 3, 4, 5, 6])

      assert_raise NimbleOptions.ValidationError,
                   "required :num_classes option not found, received options: []",
                   fn ->
                     RadiusNNClassifier.fit(x, y)
                   end
    end
  end
end
