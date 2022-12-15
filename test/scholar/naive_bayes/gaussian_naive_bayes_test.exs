defmodule Scholar.NaiveBayes.GaussianTest do
  use ExUnit.Case
  alias Scholar.NaiveBayes.Gaussian
  doctest Gaussian

  test "fit test - all default options" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model = Scholar.NaiveBayes.Gaussian.fit(x, y, num_classes: 4)

    assert model.theta ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    assert model.var ==
             Nx.tensor([
               [7.20e-08, 7.20e-08, 7.20e-08, 7.20e-08, 7.20e-08, 7.20e-08],
               [1.44e+02, 1.44e+02, 1.44e+02, 1.44e+02, 1.44e+02, 1.44e+02],
               [7.20e-08, 7.20e-08, 7.20e-08, 7.20e-08, 7.20e-08, 7.20e-08],
               [7.20e-08, 7.20e-08, 7.20e-08, 7.20e-08, 7.20e-08, 7.20e-08]
             ])

    assert model.class_priors == Nx.tensor([0.2, 0.4, 0.2, 0.2])
    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
  end

  test "fit test - :var_smoothing set to a different value" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model = Scholar.NaiveBayes.Gaussian.fit(x, y, num_classes: 4, var_smoothing: 1.0e-8)

    assert model.theta ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    assert model.var ==
             Nx.tensor([
               [
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7
               ],
               [144.0, 144.0, 144.0, 144.0, 144.0, 144.0],
               [
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7
               ],
               [
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7,
                 7.200000027296483e-7
               ]
             ])

    assert model.class_priors == Nx.tensor([0.2, 0.4, 0.2, 0.2])
    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
  end

  test "fit test - :sample_weights are not equal" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model =
      Scholar.NaiveBayes.Gaussian.fit(x, y, num_classes: 4, sample_weights: [1.5, 4, 2, 7, 4])

    assert model.theta ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [
                 17.454545974731445,
                 18.454545974731445,
                 19.454545974731445,
                 20.454545974731445,
                 21.454545974731445,
                 22.454545974731445
               ],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    assert model.var ==
             Nx.tensor([
               [
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8
               ],
               [
                 114.24793243408203,
                 114.24793243408203,
                 114.24793243408203,
                 114.24793243408203,
                 114.24793243408203,
                 114.24793243408203
               ],
               [
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8
               ],
               [
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8
               ]
             ])

    assert model.class_priors ==
             Nx.tensor([
               0.10810811072587967,
               0.29729729890823364,
               0.21621622145175934,
               0.37837839126586914
             ])

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([2.0, 5.5, 4.0, 7.0])
  end

  test "fit test - :priors are set" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model = Scholar.NaiveBayes.Gaussian.fit(x, y, num_classes: 4, priors: [0.15, 0.25, 0.4, 0.2])

    assert model.theta ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    assert model.var ==
             Nx.tensor([
               [
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8
               ],
               [144.0, 144.0, 144.0, 144.0, 144.0, 144.0],
               [
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8
               ],
               [
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8,
                 7.199999885187935e-8
               ]
             ])

    assert model.class_priors ==
             Nx.tensor([0.15000000596046448, 0.25, 0.4000000059604645, 0.20000000298023224])

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
  end

  describe "errors" do
    test "wrong input rank" do
      assert_raise ArgumentError,
                   "wrong input rank. Expected x to be rank 2 got: 1",
                   fn ->
                     Scholar.NaiveBayes.Gaussian.fit(
                       Nx.tensor([1, 2, 5, 8]),
                       Nx.tensor([1, 2, 3, 4]),
                       num_classes: 4
                     )
                   end
    end

    test "wrong target rank" do
      assert_raise ArgumentError,
                   "wrong target rank. Expected target to be rank 1 got: 2",
                   fn ->
                     Scholar.NaiveBayes.Gaussian.fit(
                       Nx.tensor([[1, 2, 5, 8]]),
                       Nx.tensor([[1, 2, 3, 4]]),
                       num_classes: 4
                     )
                   end
    end

    test "wrong input shape" do
      assert_raise ArgumentError,
                   "wrong input shape. Expected x to have the same first dimension as y, got: 1 for x and 4 for y",
                   fn ->
                     Scholar.NaiveBayes.Gaussian.fit(
                       Nx.tensor([[1, 2, 5, 8]]),
                       Nx.tensor([1, 2, 3, 4]),
                       num_classes: 4
                     )
                   end
    end

    test "wrong prior size" do
      assert_raise ArgumentError,
                   "number of priors must match number of classes. Number of priors: 3 does not match number of classes: 2",
                   fn ->
                     Scholar.NaiveBayes.Gaussian.fit(
                       Nx.tensor([[1, 2, 5, 8], [2, 5, 7, 3]]),
                       Nx.tensor([1, 0]),
                       num_classes: 2,
                       priors: [0.4, 0.4, 0.2]
                     )
                   end
    end

    test "wrong sample_weights size" do
      assert_raise ArgumentError,
                   "number of weights must match number of samples. Number of weights: 3 does not match number of samples: 2",
                   fn ->
                     Scholar.NaiveBayes.Gaussian.fit(
                       Nx.tensor([[1, 2, 5, 8], [2, 5, 7, 3]]),
                       Nx.tensor([1, 0]),
                       num_classes: 2,
                       sample_weights: [0.4, 0.4, 0.2]
                     )
                   end
    end

    test "wrong input shape in training process" do
      assert_raise ArgumentError,
                   "wrong input shape. Expected x to have the same second dimension as the data for fitting process, got: 3 for x and 4 for training data",
                   fn ->
                     model =
                       Scholar.NaiveBayes.Gaussian.fit(
                         Nx.tensor([[1, 2, 5, 8], [1, 3, 5, 2]]),
                         Nx.tensor([0, 1]),
                         num_classes: 2
                       )

                     Scholar.NaiveBayes.Gaussian.predict(model, Nx.tensor([[1, 4, 2]]))
                   end
    end
  end
end
