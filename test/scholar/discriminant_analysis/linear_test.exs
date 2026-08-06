defmodule Scholar.DiscriminantAnalysis.LinearTest do
  use Scholar.Case, async: true

  alias Scholar.DiscriminantAnalysis.Linear

  doctest Linear

  # Three well-separated classes in 2D. Reference values (coefficients,
  # intercept, decision_function and predict_proba) come from scikit-learn
  # 1.6.1 LinearDiscriminantAnalysis(solver="svd").
  defp three_class do
    x =
      Nx.tensor(
        [
          [0.0, 0.0],
          [0.5, 0.2],
          [0.2, 0.4],
          [0.1, 0.1],
          [5.0, 5.0],
          [5.2, 4.8],
          [4.8, 5.1],
          [5.1, 4.9],
          [10.0, 0.0],
          [10.2, 0.3],
          [9.8, 0.1],
          [10.1, 0.2]
        ],
        type: :f64
      )

    y = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    {x, y}
  end

  defp three_class_test_points do
    Nx.tensor([[0.3, 0.3], [5.0, 5.0], [10.0, 0.1], [2.5, 2.5]], type: :f64)
  end

  describe "fit and predict" do
    test "separates three classes" do
      {x, y} = three_class()
      model = Linear.fit(x, y, num_classes: 3)

      preds = Linear.predict(model, three_class_test_points())
      assert Nx.to_flat_list(preds) == [0, 1, 2, 0]
    end

    test "separates two classes" do
      x =
        Nx.tensor([
          [-2.0, -1.0],
          [-1.0, -1.0],
          [-1.0, -2.0],
          [1.0, 1.0],
          [1.0, 2.0],
          [2.0, 1.0]
        ])

      y = Nx.tensor([0, 0, 0, 1, 1, 1])
      model = Linear.fit(x, y, num_classes: 2)

      assert Nx.to_flat_list(Linear.predict(model, x)) == [0, 0, 0, 1, 1, 1]
    end

    test "recovers the fitted parameters" do
      {x, y} = three_class()
      model = Linear.fit(x, y, num_classes: 3)

      assert Nx.shape(model.coefficients) == {3, 2}
      assert Nx.shape(model.intercept) == {3}
      assert Nx.shape(model.means) == {3, 2}
      assert_all_close(model.priors, Nx.tensor([1 / 3, 1 / 3, 1 / 3], type: :f64))
    end
  end

  describe "matches scikit-learn" do
    test "coefficients, intercept and decision_function (f64)" do
      {x, y} = three_class()
      model = Linear.fit(x, y, num_classes: 3)

      expected_coef =
        Nx.tensor(
          [
            [-134.33268858800787, -54.50676982591884],
            [-16.508704061895486, 155.8413926499033],
            [150.8413926499034, -101.33462282398445]
          ],
          type: :f64
        )

      expected_intercept =
        Nx.tensor([406.45345089637414, -440.3788750223887, -1043.8895133202616], type: :f64)

      expected_decision =
        Nx.tensor(
          [
            [349.80161337219613, -398.57906844598637, -1029.0374823724858],
            [-537.7438411732594, 256.2845679176504, -796.3556641906669],
            [-942.3241119662964, -589.8817763763532, 454.3909508963741],
            [-65.6451951384426, -92.04715355236914, -920.1225887554642]
          ],
          type: :f64
        )

      assert Nx.type(model.coefficients) == {:f, 64}
      assert_all_close(model.coefficients, expected_coef, atol: 1.0e-6)
      assert_all_close(model.intercept, expected_intercept, atol: 1.0e-6)

      decision = Linear.decision_function(model, three_class_test_points())
      assert_all_close(decision, expected_decision, atol: 1.0e-6)
    end

    test "predict_probability (f64)" do
      {x, y} = three_class()
      model = Linear.fit(x, y, num_classes: 3)

      expected =
        Nx.tensor(
          [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.9999999999965821, 3.4180243270357275e-12, 0.0]
          ],
          type: :f64
        )

      probs = Linear.predict_probability(model, three_class_test_points())
      assert_all_close(probs, expected, atol: 1.0e-6)
    end

    # Unequal class sizes exercise the interaction between the prior term and the
    # (num_samples - num_classes) covariance normalization, which balanced
    # classes do not.
    test "coefficients, intercept and decision_function with unbalanced classes (f64)" do
      x =
        Nx.tensor(
          [
            [0.0, 0.0],
            [0.5, 0.2],
            [0.2, 0.4],
            [0.1, 0.1],
            [0.3, 0.0],
            [0.0, 0.3],
            [5.0, 5.0],
            [5.2, 4.8],
            [4.8, 5.1],
            [10.0, 0.0],
            [10.2, 0.3]
          ],
          type: :f64
        )

      y = Nx.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
      model = Linear.fit(x, y, num_classes: 3)

      assert_all_close(
        model.priors,
        Nx.tensor([6 / 11, 3 / 11, 2 / 11], type: :f64)
      )

      expected_coef =
        Nx.tensor(
          [
            [-90.99560744104983, -55.87429195011562],
            [57.705113660897624, 130.21319023823452],
            [186.42915183180307, -27.696909507004875]
          ],
          type: :f64
        )

      expected_intercept =
        Nx.tensor([203.67786828506502, -660.0225187014742, -1228.3078001516808], type: :f64)

      expected_decision =
        Nx.tensor(
          [
            [159.61689846771537, -603.6470275317345, -1180.6881274542413],
            [-530.6716286707622, 279.56900079418654, -434.6465885276898],
            [-711.8656353204449, -69.95006306867447, 633.2140272156494],
            [-163.4968801928486, -190.2267589536438, -831.4771943396853]
          ],
          type: :f64
        )

      assert_all_close(model.coefficients, expected_coef, atol: 1.0e-6)
      assert_all_close(model.intercept, expected_intercept, atol: 1.0e-6)

      decision = Linear.decision_function(model, three_class_test_points())
      assert_all_close(decision, expected_decision, atol: 1.0e-6)
      assert Nx.to_flat_list(Linear.predict(model, three_class_test_points())) == [0, 1, 2, 0]
    end
  end

  describe "properties" do
    test "keeps f64 through the model" do
      {x, y} = three_class()
      model = Linear.fit(x, y, num_classes: 3)
      assert Nx.type(model.coefficients) == {:f, 64}
      assert Nx.type(model.intercept) == {:f, 64}
    end

    test "casts integer input to f32" do
      x = Nx.tensor([[0, 0], [1, 0], [8, 8], [9, 9]])
      y = Nx.tensor([0, 0, 1, 1])
      model = Linear.fit(x, y, num_classes: 2)
      assert Nx.type(model.coefficients) == {:f, 32}
      assert Nx.to_flat_list(Linear.predict(model, x)) == [0, 0, 1, 1]
    end

    test "works inside jit" do
      {x, y} = three_class()
      xt = three_class_test_points()

      direct = Linear.fit(x, y, num_classes: 3) |> Linear.predict(xt)

      jitted =
        Nx.Defn.jit_apply(
          fn x, y, xt -> Linear.fit(x, y, num_classes: 3) |> Linear.predict(xt) end,
          [x, y, xt]
        )

      assert Nx.to_flat_list(jitted) == Nx.to_flat_list(direct)
    end
  end

  describe "errors" do
    test "raises for a non-rank-2 x" do
      assert_raise ArgumentError,
                   "expected x to have shape {num_samples, num_features}, " <>
                     "got tensor with shape: {3}",
                   fn ->
                     Linear.fit(Nx.tensor([1, 2, 3]), Nx.tensor([0, 1, 0]), num_classes: 2)
                   end
    end

    test "raises for a non-rank-1 y" do
      assert_raise ArgumentError,
                   "expected y to have shape {num_samples}, " <>
                     "got tensor with shape: {2, 1}",
                   fn ->
                     Linear.fit(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), Nx.tensor([[0], [1]]),
                       num_classes: 2
                     )
                   end
    end

    test "raises when x and y have different number of samples" do
      assert_raise ArgumentError,
                   "expected x and y to have the same number of samples, got 2 and 3",
                   fn ->
                     Linear.fit(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), Nx.tensor([0, 1, 0]),
                       num_classes: 2
                     )
                   end
    end

    test "raises when num_classes is missing" do
      assert_raise NimbleOptions.ValidationError, fn ->
        Linear.fit(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), Nx.tensor([0, 1]), [])
      end
    end
  end
end
