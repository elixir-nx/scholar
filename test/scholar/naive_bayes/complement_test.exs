defmodule Scholar.NaiveBayes.ComplementTest do
  use ExUnit.Case
  import ScholarCase
  alias Scholar.NaiveBayes.Complement
  doctest Complement

  test "fit test - all default options" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model = Scholar.NaiveBayes.Complement.fit(x, y, num_classes: 4)

    assert model.feature_count ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    expected_feature_log_probability =
      Nx.tensor([
        [
          1.9774765968322754,
          1.8990050554275513,
          1.8262455463409424,
          1.758423089981079,
          1.694909691810608,
          1.635190486907959
        ],
        [
          1.9763307571411133,
          1.8983691930770874,
          1.826048493385315,
          1.758607268333435,
          1.6954283714294434,
          1.63600492477417
        ],
        [
          1.9588135480880737,
          1.8886092901229858,
          1.8230119943618774,
          1.7614541053771973,
          1.7034668922424316,
          1.6486586332321167
        ],
        [
          2.0008511543273926,
          1.911903738975525,
          1.8302258253097534,
          1.7547181844711304,
          1.6845139265060425,
          1.618916630744934
        ]
      ])

    assert_all_close(model.feature_log_probability, expected_feature_log_probability)

    expected_class_log_priors =
      Nx.tensor([
        -1.6094379425048828,
        -0.9162907600402832,
        -1.6094379425048828,
        -1.6094379425048828
      ])

    assert_all_close(model.class_log_priors, expected_class_log_priors)

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
    assert model.feature_all == Nx.tensor([60.0, 65.0, 70.0, 75.0, 80.0, 85.0])
  end

  test "fit test - :alpha set to a different value" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model = Scholar.NaiveBayes.Complement.fit(x, y, num_classes: 4, alpha: 1.0e-6)

    assert model.feature_count ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    expected_feature_log_probability =
      Nx.tensor([
        [
          1.9810014963150024,
          1.900958776473999,
          1.8268507719039917,
          1.7578579187393188,
          1.6933194398880005,
          1.6326948404312134
        ],
        [
          1.9810014963150024,
          1.900958776473999,
          1.8268507719039917,
          1.7578579187393188,
          1.6933194398880005,
          1.6326948404312134
        ],
        [
          1.9616584777832031,
          1.8901995420455933,
          1.8235081434249878,
          1.7609877586364746,
          1.7021472454071045,
          1.6465774774551392
        ],
        [
          2.005333423614502,
          1.9143617153167725,
          1.8309801816940308,
          1.7540191411972046,
          1.6825602054595947,
          1.6158688068389893
        ]
      ])

    assert_all_close(model.feature_log_probability, expected_feature_log_probability)

    expected_class_log_priors =
      Nx.tensor([
        -1.6094379425048828,
        -0.9162907600402832,
        -1.6094379425048828,
        -1.6094379425048828
      ])

    assert_all_close(model.class_log_priors, expected_class_log_priors)

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
    assert model.feature_all == Nx.tensor([60.0, 65.0, 70.0, 75.0, 80.0, 85.0])
  end

  test "fit test - :fit_priors set to false" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model = Scholar.NaiveBayes.Complement.fit(x, y, num_classes: 4, fit_priors: false)

    assert model.feature_count ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    expected_feature_log_probability =
      Nx.tensor([
        [
          1.9774765968322754,
          1.8990050554275513,
          1.8262455463409424,
          1.758423089981079,
          1.694909691810608,
          1.635190486907959
        ],
        [
          1.9763307571411133,
          1.8983691930770874,
          1.826048493385315,
          1.758607268333435,
          1.6954283714294434,
          1.63600492477417
        ],
        [
          1.9588135480880737,
          1.8886092901229858,
          1.8230119943618774,
          1.7614541053771973,
          1.7034668922424316,
          1.6486586332321167
        ],
        [
          2.0008511543273926,
          1.911903738975525,
          1.8302258253097534,
          1.7547181844711304,
          1.6845139265060425,
          1.618916630744934
        ]
      ])

    assert_all_close(model.feature_log_probability, expected_feature_log_probability)

    expected_class_log_priors =
      Nx.tensor([
        -1.3862943649291992,
        -1.3862943649291992,
        -1.3862943649291992,
        -1.3862943649291992
      ])

    assert_all_close(model.class_log_priors, expected_class_log_priors)

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
    assert model.feature_all == Nx.tensor([60.0, 65.0, 70.0, 75.0, 80.0, 85.0])
  end

  test "fit test - :priors are set" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model =
      Scholar.NaiveBayes.Complement.fit(x, y, num_classes: 4, priors: [0.15, 0.25, 0.4, 0.2])

    assert model.feature_count ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    expected_feature_log_probability =
      Nx.tensor([
        [
          1.9774765968322754,
          1.8990050554275513,
          1.8262455463409424,
          1.758423089981079,
          1.694909691810608,
          1.635190486907959
        ],
        [
          1.9763307571411133,
          1.8983691930770874,
          1.826048493385315,
          1.758607268333435,
          1.6954283714294434,
          1.63600492477417
        ],
        [
          1.9588135480880737,
          1.8886092901229858,
          1.8230119943618774,
          1.7614541053771973,
          1.7034668922424316,
          1.6486586332321167
        ],
        [
          2.0008511543273926,
          1.911903738975525,
          1.8302258253097534,
          1.7547181844711304,
          1.6845139265060425,
          1.618916630744934
        ]
      ])

    assert_all_close(model.feature_log_probability, expected_feature_log_probability)

    expected_class_log_priors =
      Nx.tensor([
        -1.8971199989318848,
        -1.3862943649291992,
        -0.9162907004356384,
        -1.6094379425048828
      ])

    assert_all_close(model.class_log_priors, expected_class_log_priors)

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
    assert model.feature_all == Nx.tensor([60.0, 65.0, 70.0, 75.0, 80.0, 85.0])
  end

  test "fit test - :sample_weights are set" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model =
      Scholar.NaiveBayes.Complement.fit(x, y, num_classes: 4, sample_weights: [1.5, 4, 2, 7, 4])

    assert model.feature_count ==
             Nx.tensor([
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [96.0, 101.5, 107.0, 112.5, 118.0, 123.5],
               [24.0, 28.0, 32.0, 36.0, 40.0, 44.0],
               [126.0, 133.0, 140.0, 147.0, 154.0, 161.0]
             ])

    expected_feature_log_probability =
      Nx.tensor([
        [
          1.9461992979049683,
          1.881534218788147,
          1.8207980394363403,
          1.7635403871536255,
          1.7093844413757324,
          1.6580113172531128
        ],
        [
          1.9621047973632812,
          1.890448808670044,
          1.823585867881775,
          1.7609148025512695,
          1.7019407749176025,
          1.646251916885376
        ],
        [
          1.9287010431289673,
          1.8716551065444946,
          1.8176884651184082,
          1.7664858102798462,
          1.7177776098251343,
          1.6713320016860962
        ],
        [
          1.9726431369781494,
          1.8963209390640259,
          1.8254129886627197,
          1.7592017650604248,
          1.697103500366211,
          1.6386370658874512
        ]
      ])

    assert_all_close(model.feature_log_probability, expected_feature_log_probability)

    expected_class_log_priors =
      Nx.tensor([
        -2.224623441696167,
        -1.2130225896835327,
        -1.5314762592315674,
        -0.9718605279922485
      ])

    assert_all_close(model.class_log_priors, expected_class_log_priors)

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([2.0, 5.5, 4.0, 7.0])
    assert model.feature_all == Nx.tensor([270.0, 288.5, 307.0, 325.5, 344.0, 362.5])
  end

  test "fit test - :norm set to true" do
    x = Nx.iota({5, 6})
    y = Nx.tensor([1, 2, 0, 3, 1])

    model = Scholar.NaiveBayes.Complement.fit(x, y, num_classes: 4, norm: true)

    assert model.feature_count ==
             Nx.tensor([
               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
               [24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
               [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
             ])

    expected_feature_log_probability =
      Nx.tensor([
        [
          0.1832481473684311,
          0.17597636580467224,
          0.16923391819000244,
          0.16294896602630615,
          0.1570633351802826,
          0.15152929723262787
        ],
        [
          0.1831497997045517,
          0.17592497169971466,
          0.169222891330719,
          0.16297300159931183,
          0.15711811184883118,
          0.15161125361919403
        ],
        [
          0.18164047598838806,
          0.1751304417848587,
          0.16904762387275696,
          0.16333936154842377,
          0.15796221792697906,
          0.15287986397743225
        ],
        [
          0.185244619846344,
          0.17700961232185364,
          0.1694476306438446,
          0.1624569147825241,
          0.1559572070837021,
          0.14988401532173157
        ]
      ])

    assert_all_close(model.feature_log_probability, expected_feature_log_probability)

    expected_class_log_priors =
      Nx.tensor([
        -1.6094379425048828,
        -0.9162907600402832,
        -1.6094379425048828,
        -1.6094379425048828
      ])

    assert_all_close(model.class_log_priors, expected_class_log_priors)

    assert model.classes == Nx.tensor([0, 1, 2, 3])
    assert model.class_count == Nx.tensor([1.0, 2.0, 1.0, 1.0])
    assert model.feature_all == Nx.tensor([60.0, 65.0, 70.0, 75.0, 80.0, 85.0])
  end

  describe "errors" do
    test "wrong input rank" do
      assert_raise ArgumentError,
                   "wrong input rank. Expected x to be rank 2 got: 1",
                   fn ->
                     Scholar.NaiveBayes.Complement.fit(
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
                     Scholar.NaiveBayes.Complement.fit(
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
                     Scholar.NaiveBayes.Complement.fit(
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
                     Scholar.NaiveBayes.Complement.fit(
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
                     Scholar.NaiveBayes.Complement.fit(
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
                     Scholar.NaiveBayes.Complement.fit(
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
                       Scholar.NaiveBayes.Complement.fit(
                         Nx.tensor([[1, 2, 5, 8], [1, 3, 5, 2]]),
                         Nx.tensor([0, 1]),
                         num_classes: 2
                       )

                     Scholar.NaiveBayes.Complement.predict(model, Nx.tensor([[1, 4, 2]]))
                   end
    end
  end
end
