defmodule Scholar.NaiveBayes.BernoulliTest do
  use Scholar.Case, async: true
  alias Scholar.NaiveBayes.Bernoulli
  doctest Bernoulli

  describe "fit" do
    test "binary y" do
      x = Nx.iota({5, 6})
      x = Scholar.Preprocessing.Binarizer.fit_transform(x)
      y = Nx.tensor([1, 0, 1, 0, 1])

      model = Bernoulli.fit(x, y, num_classes: 2, binarize: nil)

      assert model.feature_count ==
               Nx.tensor([
                 [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                 [2.0, 3.0, 3.0, 3.0, 3.0, 3.0]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [-0.28768207, -0.28768207, -0.28768207, -0.28768207, -0.28768207, -0.28768207],
          [-0.51082562, -0.22314355, -0.22314355, -0.22314355, -0.22314355, -0.22314355]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([
          -0.91629073,
          -0.51082562
        ])

      assert_all_close(model.class_log_priors, expected_class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 3.0])
    end

    test ":alpha set to a different value" do
      x = Nx.iota({5, 6})
      y = Nx.tensor([1, 2, 6, 3, 1])

      model = Bernoulli.fit(x, y, num_classes: 4, alpha: 0.4)

      assert model.feature_count ==
               Nx.tensor([
                 [1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [-0.69314718, -0.15415068, -0.15415068, -0.15415068, -0.15415068, -0.15415068],
          [-0.25131443, -0.25131443, -0.25131443, -0.25131443, -0.25131443, -0.25131443],
          [-0.25131443, -0.25131443, -0.25131443, -0.25131443, -0.25131443, -0.25131443],
          [-0.25131443, -0.25131443, -0.25131443, -0.25131443, -0.25131443, -0.25131443]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-0.91629073, -1.60943791, -1.60943791, -1.60943791])

      assert_all_close(model.class_log_priors, expected_class_log_priors)
      assert_all_close(expected_class_log_priors, model.class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 1.0, 1.0, 1.0])
    end

    test ":fit_priors set to false" do
      x = Nx.iota({5, 6})
      y = Nx.tensor([1, 0, 1, 0, 1])

      model = Bernoulli.fit(x, y, num_classes: 2, fit_priors: false)

      assert model.feature_count ==
               Nx.tensor([
                 [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                 [2.0, 3.0, 3.0, 3.0, 3.0, 3.0]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [-0.28768207, -0.28768207, -0.28768207, -0.28768207, -0.28768207, -0.28768207],
          [-0.51082562, -0.22314355, -0.22314355, -0.22314355, -0.22314355, -0.22314355]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-0.69314718, -0.69314718])

      assert_all_close(model.class_log_priors, expected_class_log_priors)
      assert_all_close(expected_class_log_priors, model.class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 3.0])
    end

    #
    test "fit test - :class_priors are set as a list" do
      x = Nx.iota({5, 6})
      y = Nx.tensor([1, 2, 3, 2, 1])

      model = Bernoulli.fit(x, y, num_classes: 3, class_priors: [0.3, 0.4, 0.3])

      assert model.feature_count ==
               Nx.tensor([
                 [1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                 [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [-0.69314718, -0.28768207, -0.28768207, -0.28768207, -0.28768207, -0.28768207],
          [-0.28768207, -0.28768207, -0.28768207, -0.28768207, -0.28768207, -0.28768207],
          [-0.40546511, -0.40546511, -0.40546511, -0.40546511, -0.40546511, -0.40546511]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-1.2039728, -0.91629073, -1.2039728])

      assert_all_close(model.class_log_priors, expected_class_log_priors)
      assert_all_close(expected_class_log_priors, model.class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 2.0, 1.0])
    end

    test "error handling for wrong input shapes" do
      assert_raise ArgumentError,
                   "expected x to have shape {num_samples, num_features}, got tensor with shape: {4}",
                   fn ->
                     Bernoulli.fit(
                       Nx.tensor([1, 2, 3, 4]),
                       Nx.tensor([1, 0, 1, 0]),
                       num_classes: 2
                     )
                   end

      assert_raise ArgumentError,
                   "expected y to have shape {num_samples}, got tensor with shape: {1, 4}",
                   fn ->
                     Bernoulli.fit(
                       Nx.tensor([[1, 2, 3, 4]]),
                       Nx.tensor([[1, 0, 1, 0]]),
                       num_classes: 2
                     )
                   end
    end
  end

  describe "predict" do
    test "predicts classes correctly for new data" do
      x = Nx.iota({5, 6})
      y = Nx.tensor([1, 2, 3, 4, 5])

      jit_model = Nx.Defn.jit(&Bernoulli.fit/3)
      model = jit_model.(x, y, num_classes: 5)

      x_test = Nx.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]])

      jit_predict = Nx.Defn.jit(&Bernoulli.predict/3)
      predictions = jit_predict.(model, x_test, Nx.tensor([1, 2, 3, 4, 5]))

      expected_predictions = Nx.tensor([2, 1])
      assert predictions == expected_predictions
    end

    test "applies binarize to the input at predict time, not only at fit time" do
      x = Nx.iota({4, 3})
      y = Nx.tensor([1, 2, 0, 2])
      model = Bernoulli.fit(x, y, num_classes: 3)

      x_test = Nx.tensor([[6, 2, 4], [8, 5, 9]])

      # Every entry of x_test is above the default threshold of 0.0, so both
      # rows binarize to all-ones and must therefore score identically.
      # Reference: sklearn.naive_bayes.BernoulliNB(binarize=0.0) on the same
      # data returns these probabilities for both rows.
      assert_all_close(
        Bernoulli.predict_probability(model, x_test),
        Nx.tensor([
          [0.23000899, 0.11500449, 0.65498656],
          [0.23000899, 0.11500449, 0.65498656]
        ])
      )

      # Passing the already-binarized input must give the same answer.
      x_test_binarized = Scholar.Preprocessing.Binarizer.fit_transform(x_test, threshold: 0.0)
      model_no_binarize = Bernoulli.fit(x, y, num_classes: 3)

      assert_all_close(
        Bernoulli.predict_probability(model_no_binarize, x_test_binarized),
        Bernoulli.predict_probability(model, x_test)
      )
    end

    test "respects a non-default binarize threshold at predict time" do
      x = Nx.iota({4, 3})
      y = Nx.tensor([1, 2, 0, 2])
      model = Bernoulli.fit(x, y, num_classes: 3, binarize: 5.0)

      x_test = Nx.tensor([[6, 2, 4], [8, 5, 9]])

      # With threshold 5.0 the two rows binarize differently ([1,0,0] and
      # [1,0,1]), so unlike the default-threshold case they must not score
      # identically.
      probability = Bernoulli.predict_probability(model, x_test)
      refute Nx.to_number(Nx.all_close(probability[0], probability[1])) == 1
    end
  end

  describe "option validation" do
    test "rejects a negative alpha instead of silently producing NaN" do
      x = Nx.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
      y = Nx.tensor([0, 1, 1])

      assert_raise NimbleOptions.ValidationError, fn ->
        Bernoulli.fit(x, y, num_classes: 2, alpha: -1.0)
      end

      assert_raise NimbleOptions.ValidationError, fn ->
        Bernoulli.fit(x, y, num_classes: 2, alpha: [1.0, -2.0, 1.0])
      end
    end

    test "keeps class_log_priors in the same type as the rest of the model" do
      x = Nx.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], type: :f64)
      y = Nx.tensor([0, 1, 1])

      for opts <- [
            [num_classes: 2],
            [num_classes: 2, class_priors: [0.3, 0.7]],
            [num_classes: 2, fit_priors: false]
          ] do
        model = Bernoulli.fit(x, y, opts)

        assert Nx.type(model.class_log_priors) == Nx.type(model.feature_log_probability),
               "class_log_priors downcast for opts: #{inspect(opts)}"
      end
    end

    test "computes the uniform prior in the model type, not in f32" do
      x = Nx.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], type: :f64)
      y = Nx.tensor([0, 1, 1])
      model = Bernoulli.fit(x, y, num_classes: 3, fit_priors: false)

      # -log(3) must be accurate to f64, not an f32 value widened to f64.
      assert_all_close(model.class_log_priors[0], Nx.tensor(-:math.log(3), type: :f64),
        atol: 1.0e-15
      )
    end
  end
end
