defmodule Scholar.NaiveBayes.MultinomialTest do
  use ExUnit.Case
  alias Scholar.NaiveBayes.Multinomial
  doctest Multinomial

  test "fit test - all default options" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model = Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 4)

    assert model.feature_count ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    assert model.feature_log_probability ==
             Nx.tensor([
               [
                 -1.9676501750946045,
                 -1.8935420513153076,
                 -1.8245491981506348,
                 -1.7600107192993164,
                 -1.6993861198425293,
                 -1.6422276496887207
               ],
               [
                 -1.974081039428711,
                 -1.8971199989318848,
                 -1.8256611824035645,
                 -1.758969783782959,
                 -1.6964492797851562,
                 -1.6376087665557861
               ],
               [
                 -2.0971412658691406,
                 -1.9636096954345703,
                 -1.8458266258239746,
                 -1.7404661178588867,
                 -1.645155906677246,
                 -1.5581445693969727
               ],
               [
                 -1.9153733253479004,
                 -1.8640799522399902,
                 -1.8152897357940674,
                 -1.7687697410583496,
                 -1.724318027496338,
                 -1.6817584037780762
               ]
             ])

    assert model.class_log_priors ==
             Nx.tensor([
               -1.6094379425048828,
               -0.9162907600402832,
               -1.6094379425048828,
               -1.6094379425048828
             ])

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
  end

  test "fit test - :alpha set to a different value" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model = Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 4, alpha: 1.0e-6)

    assert model.feature_count ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    assert model.feature_log_probability ==
             Nx.tensor([
               [
                 -1.981001377105713,
                 -1.90095853805542,
                 -1.8268506526947021,
                 -1.7578577995300293,
                 -1.693319320678711,
                 -1.6326944828033447
               ],
               [
                 -1.981001377105713,
                 -1.90095853805542,
                 -1.8268506526947021,
                 -1.7578577995300293,
                 -1.693319320678711,
                 -1.6326947212219238
               ],
               [
                 -2.140066146850586,
                 -1.9859155416488647,
                 -1.852384328842163,
                 -1.7346012592315674,
                 -1.6292407512664795,
                 -1.5339305400848389
               ],
               [
                 -1.9218125343322754,
                 -1.8677451610565186,
                 -1.8164520263671875,
                 -1.7676618099212646,
                 -1.7211418151855469,
                 -1.6766901016235352
               ]
             ])

    assert model.class_log_priors ==
             Nx.tensor([
               -1.6094379425048828,
               -0.9162907600402832,
               -1.6094379425048828,
               -1.6094379425048828
             ])

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
  end

  test "fit test - :fit_priors set to false" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model = Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 4, fit_priors: false)

    assert model.feature_count ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    assert model.feature_log_probability ==
             Nx.tensor([
               [
                 -1.9676501750946045,
                 -1.8935420513153076,
                 -1.8245491981506348,
                 -1.7600107192993164,
                 -1.6993861198425293,
                 -1.6422276496887207
               ],
               [
                 -1.974081039428711,
                 -1.8971199989318848,
                 -1.8256611824035645,
                 -1.758969783782959,
                 -1.6964492797851562,
                 -1.6376087665557861
               ],
               [
                 -2.0971412658691406,
                 -1.9636096954345703,
                 -1.8458266258239746,
                 -1.7404661178588867,
                 -1.645155906677246,
                 -1.5581445693969727
               ],
               [
                 -1.9153733253479004,
                 -1.8640799522399902,
                 -1.8152897357940674,
                 -1.7687697410583496,
                 -1.724318027496338,
                 -1.6817584037780762
               ]
             ])

    assert model.class_log_priors ==
             Nx.tensor([
               -1.3862943649291992,
               -1.3862943649291992,
               -1.3862943649291992,
               -1.3862943649291992
             ])

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
  end

  test "fit test - :priors are set" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model =
      Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 4, priors: [0.15, 0.25, 0.4, 0.2])

    assert model.feature_count ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    assert model.feature_log_probability ==
             Nx.tensor([
               [
                 -1.9676501750946045,
                 -1.8935420513153076,
                 -1.8245491981506348,
                 -1.7600107192993164,
                 -1.6993861198425293,
                 -1.6422276496887207
               ],
               [
                 -1.974081039428711,
                 -1.8971199989318848,
                 -1.8256611824035645,
                 -1.758969783782959,
                 -1.6964492797851562,
                 -1.6376087665557861
               ],
               [
                 -2.0971412658691406,
                 -1.9636096954345703,
                 -1.8458266258239746,
                 -1.7404661178588867,
                 -1.645155906677246,
                 -1.5581445693969727
               ],
               [
                 -1.9153733253479004,
                 -1.8640799522399902,
                 -1.8152897357940674,
                 -1.7687697410583496,
                 -1.724318027496338,
                 -1.6817584037780762
               ]
             ])

    assert model.class_log_priors ==
             Nx.tensor([
               -1.8971199989318848,
               -1.3862943649291992,
               -0.9162907004356384,
               -1.6094379425048828
             ])

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
  end

  test "fit test - :sample_weights are set" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model =
      Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 4, sample_weights: [1.5, 4, 2, 7, 4])

    assert model.feature_count ==
             Nx.tensor([
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [96.0, 101.5, 107.0, 112.5, 118.0, 123.5],
               [24.0, 28.0, 32.0, 36.0, 40.0, 44.0],
               [126.0, 133.0, 140.0, 147.0, 154.0, 161.0]
             ])

    assert model.feature_log_probability ==
             Nx.tensor([
               [
                 -1.974081039428711,
                 -1.8971199989318848,
                 -1.8256611824035645,
                 -1.758969783782959,
                 -1.6964492797851562,
                 -1.6376087665557861
               ],
               [
                 -1.9243240356445312,
                 -1.8691720962524414,
                 -1.8169035911560059,
                 -1.7672319412231445,
                 -1.7199115753173828,
                 -1.674729347229004
               ],
               [
                 -2.1282315254211426,
                 -1.979811668395996,
                 -1.850599765777588,
                 -1.736189603805542,
                 -1.633535385131836,
                 -1.5404448509216309
               ],
               [
                 -1.920851707458496,
                 -1.8671989440917969,
                 -1.8162789344787598,
                 -1.7678265571594238,
                 -1.721613883972168,
                 -1.6774425506591797
               ]
             ])

    assert model.class_log_priors ==
             Nx.tensor([
               -2.224623441696167,
               -1.2130225896835327,
               -1.5314762592315674,
               -0.9718605279922485
             ])

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([2.0, 5.5, 4.0, 7.0])
  end

  describe "errors" do
    test "wrong input rank" do
      assert_raise ArgumentError,
                   "wrong input rank. Expected x to be rank 2 got: 1",
                   fn ->
                     Scholar.NaiveBayes.Multinomial.fit(
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
                     Scholar.NaiveBayes.Multinomial.fit(
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
                     Scholar.NaiveBayes.Multinomial.fit(
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
                     Scholar.NaiveBayes.Multinomial.fit(
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
                     Scholar.NaiveBayes.Multinomial.fit(
                       Nx.tensor([[1, 2, 5, 8], [2, 5, 7, 3]]),
                       Nx.tensor([1, 0]),
                       num_classes: 2,
                       sample_weights: [0.4, 0.4, 0.2]
                     )
                   end
    end

    test "wrong alpha size" do
      assert_raise ArgumentError,
                   "when alpha is a list it should contain num_features values",
                   fn ->
                     Scholar.NaiveBayes.Multinomial.fit(
                       Nx.tensor([[1, 2, 5, 8], [2, 5, 7, 3]]),
                       Nx.tensor([1, 0]),
                       num_classes: 2,
                       alpha: [0.4, 0.4, 0.2]
                     )
                   end
    end

    test "wrong input shape in training process" do
      assert_raise ArgumentError,
                   "wrong input shape. Expected x to have the same second dimension as the data for fitting process, got: 3 for x and 4 for training data",
                   fn ->
                     model =
                       Scholar.NaiveBayes.Multinomial.fit(
                         Nx.tensor([[1, 2, 5, 8], [1, 3, 5, 2]]),
                         Nx.tensor([0, 1]),
                         num_classes: 2
                       )

                     Scholar.NaiveBayes.Multinomial.predict(model, Nx.tensor([[1, 4, 2]]))
                   end
    end
  end
end
