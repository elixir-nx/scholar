defmodule Scholar.Linear.RidgeRegressionTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.RidgeRegression
  doctest RidgeRegression

  describe "fit" do
    test "solver - :svd, options set to default, and x with shape num_samples >= num_features" do
      x = Nx.tensor([[1, 6], [2, 7], [9, 5]])
      y = Nx.tensor([1, 5, 2])

      expected_coeff = Nx.tensor([0.17647059, 1.41176471])
      expected_intercept = Nx.tensor(-6.509803921568629)

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y, solver: :svd)

      assert_all_close(expected_coeff, actual_coeff, atol: 1.0e-3, rtol: 1.0e-3)
      assert_all_close(expected_intercept, actual_intercept, atol: 1.0e-3, rtol: 1.0e-3)
    end

    test "solver - :svd, options set to default, and x with shape num_samples < num_features" do
      x = Nx.tensor([[1, 6, 3, 4], [2, 7, 5, 6], [9, 5, 7, 8]])
      y = Nx.tensor([1, 5, 2])

      expected_coeff = Nx.tensor([-0.49019608, 0.74509804, 0.66666667, 0.66666667])
      expected_intercept = Nx.tensor(-7.176470588235292)

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y, solver: :svd)

      assert_all_close(expected_coeff, actual_coeff, atol: 1.0e-3, rtol: 1.0e-3)
      assert_all_close(expected_intercept, actual_intercept, atol: 1.0e-3, rtol: 1.0e-3)
    end

    test "solver - :svd, alpha as a scalar, and x with shape num_samples < num_features" do
      x = Nx.tensor([[1, 6, 3, 4], [2, 7, 5, 6], [9, 5, 7, 8]])
      y = Nx.tensor([1, 5, 2])

      expected_coeff = Nx.tensor([-0.46651827, 0.70766092, 0.63253571, 0.63253571])
      expected_intercept = Nx.tensor(-6.671118530884806)

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y, alpha: 1.2, solver: :svd)

      assert_all_close(expected_coeff, actual_coeff, atol: 1.0e-3, rtol: 1.0e-3)
      assert_all_close(expected_intercept, actual_intercept, atol: 1.0e-3, rtol: 1.0e-3)
    end

    test "solver - :svd, alpha as a scalar, weights as a list, and x with shape num_samples < num_features" do
      x = Nx.tensor([[1, 6, 3, 4], [2, 7, 5, 6], [9, 5, 7, 8]])
      y = Nx.tensor([1, 5, 2])

      expected_coeff = Nx.tensor([-0.33184292, 0.46423017, 0.39774495, 0.39774495])
      expected_intercept = Nx.tensor(-3.088452566096427)

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y,
          alpha: 1.2,
          sample_weights: [0.3, 0.4, 0.5],
          solver: :svd
        )

      assert_all_close(expected_coeff, actual_coeff, atol: 1.0e-3, rtol: 1.0e-3)
      assert_all_close(expected_intercept, actual_intercept, atol: 1.0e-3, rtol: 1.0e-2)
    end

    test "solver - :svd, alpha as a list, weights as a list, and x with shape num_samples < num_features" do
      x = Nx.tensor([[1, 6, 3, 4], [2, 7, 5, 6], [9, 5, 7, 8]])
      y = Nx.tensor([[1, 2], [5, 7], [2, 5]])

      expected_coeff =
        Nx.tensor([
          [-0.33184292, 0.46423017, 0.39774495, 0.39774495],
          [-0.14960991, 0.36668196, 0.38916934, 0.38916934]
        ])

      expected_intercept = Nx.tensor([-3.08845257, -1.09499771])

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y,
          alpha: [1.2, 2.0],
          sample_weights: [0.3, 0.4, 0.5],
          solver: :svd
        )

      assert_all_close(expected_coeff, actual_coeff, atol: 1.0e-3, rtol: 1.0e-3)
      assert_all_close(expected_intercept, actual_intercept, atol: 1.0e-3, rtol: 1.0e-2)
    end

    test "solver - :svd, alpha as a list, weights as a list, and x with shape num_samples >= num_features" do
      x = Nx.tensor([[1, 6], [2, 7], [9, 5]])
      y = Nx.tensor([[1, 2], [5, 7], [2, 5]])

      expected_coeff =
        Nx.tensor([
          [-0.01035276, 0.59355828],
          [0.14004721, 0.43036979]
        ])

      expected_intercept = Nx.tensor([-0.71357362, 1.71675846])

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y,
          alpha: [1.2, 2.0],
          sample_weights: [0.3, 0.4, 0.5],
          solver: :svd
        )

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "solver - :svd, alpha as a tensor, weights as a tensor, and x with shape num_samples >= num_features" do
      x = Nx.tensor([[1, 6], [2, 7], [9, 5]])
      y = Nx.tensor([[1, 2], [5, 7], [2, 5]])

      expected_coeff =
        Nx.tensor([
          [-0.01035276, 0.59355828],
          [0.14004721, 0.43036979]
        ])

      expected_intercept = Nx.tensor([-0.71357362, 1.71675846])

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y,
          alpha: Nx.tensor([1.2, 2.0]),
          sample_weights: Nx.tensor([0.3, 0.4, 0.5]),
          solver: :svd
        )

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "solver - :cholesky, options set to default, and x with shape num_samples >= num_features" do
      x = Nx.tensor([[1, 6], [2, 7], [9, 5]])
      y = Nx.tensor([1, 5, 2])

      expected_coeff = Nx.tensor([0.17647059, 1.41176471])
      expected_intercept = Nx.tensor(-6.509803921568629)

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y, solver: :cholesky)

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "solver - :cholesky, options set to default, and x with shape num_samples < num_features" do
      x = Nx.tensor([[1, 6, 3, 4], [2, 7, 5, 6], [9, 5, 7, 8]])
      y = Nx.tensor([1, 5, 2])

      expected_coeff = Nx.tensor([-0.49019608, 0.74509804, 0.66666667, 0.66666667])
      expected_intercept = Nx.tensor(-7.176470588235292)

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y, solver: :cholesky)

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "solver - :cholesky, alpha as a scalar, and x with shape num_samples < num_features" do
      x = Nx.tensor([[1, 6, 3, 4], [2, 7, 5, 6], [9, 5, 7, 8]])
      y = Nx.tensor([1, 5, 2])

      expected_coeff = Nx.tensor([-0.46651827, 0.70766092, 0.63253571, 0.63253571])
      expected_intercept = Nx.tensor(-6.671118530884806)

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y, alpha: 1.2, solver: :cholesky)

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "solver - :cholesky, alpha as a scalar, weights as a list, and x with shape num_samples < num_features" do
      x = Nx.tensor([[1, 6, 3, 4], [2, 7, 5, 6], [9, 5, 7, 8]])
      y = Nx.tensor([1, 5, 2])

      expected_coeff = Nx.tensor([-0.33184292, 0.46423017, 0.39774495, 0.39774495])
      expected_intercept = Nx.tensor(-3.088452566096427)

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y,
          alpha: 1.2,
          sample_weights: [0.3, 0.4, 0.5],
          solver: :cholesky
        )

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "solver - :cholesky, alpha as a list, weights as a list, and x with shape num_samples < num_features" do
      x = Nx.tensor([[1, 6, 3, 4], [2, 7, 5, 6], [9, 5, 7, 8]])
      y = Nx.tensor([[1, 2], [5, 7], [2, 5]])

      expected_coeff =
        Nx.tensor([
          [-0.33184292, 0.46423017, 0.39774495, 0.39774495],
          [-0.14960991, 0.36668196, 0.38916934, 0.38916934]
        ])

      expected_intercept = Nx.tensor([-3.08845257, -1.09499771])

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y,
          alpha: [1.2, 2.0],
          sample_weights: [0.3, 0.4, 0.5],
          solver: :cholesky
        )

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "solver - :cholesky, alpha as a list, weights as a list, and x with shape num_samples >= num_features" do
      x = Nx.tensor([[1, 6], [2, 7], [9, 5]])
      y = Nx.tensor([[1, 2], [5, 7], [2, 5]])

      expected_coeff =
        Nx.tensor([
          [-0.01035276, 0.59355828],
          [0.14004721, 0.43036979]
        ])

      expected_intercept = Nx.tensor([-0.71357362, 1.71675846])

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y,
          alpha: [1.2, 2.0],
          sample_weights: [0.3, 0.4, 0.5],
          solver: :cholesky
        )

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "solver - :cholesky, alpha as a tensor, weights as a tensor, and x with shape num_samples >= num_features" do
      x = Nx.tensor([[1, 6], [2, 7], [9, 5]])
      y = Nx.tensor([[1, 2], [5, 7], [2, 5]])

      expected_coeff =
        Nx.tensor([
          [-0.01035276, 0.59355828],
          [0.14004721, 0.43036979]
        ])

      expected_intercept = Nx.tensor([-0.71357362, 1.71675846])

      %RidgeRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        RidgeRegression.fit(x, y,
          alpha: Nx.tensor([1.2, 2.0]),
          sample_weights: Nx.tensor([0.3, 0.4, 0.5]),
          solver: :cholesky
        )

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end
  end

  describe "predict" do
    test "solver - :cholesky, alpha as a tensor, weights as a tensor, and x with shape num_samples >= num_features" do
      x = Nx.tensor([[1, 6], [2, 7], [9, 5]])
      y = Nx.tensor([[1, 2], [5, 7], [2, 5]])

      model =
        RidgeRegression.fit(x, y,
          alpha: Nx.tensor([1.2, 2.0]),
          sample_weights: Nx.tensor([0.3, 0.4, 0.5]),
          solver: :cholesky
        )

      to_predict = Nx.tensor([[4, 17]])
      expected_prediction = Nx.tensor([[9.33550613, 9.59323367]])
      actual_prediction = RidgeRegression.predict(model, to_predict)
      assert_all_close(expected_prediction, actual_prediction)
    end
  end

  @tag :wip
  test "toy ridge with column target" do
    x = Nx.tensor([[1], [2], [6], [8], [10]])
    y = Nx.tensor([1, 2, 6, 8, 10])
    model = RidgeRegression.fit(x, y)
    pred = RidgeRegression.predict(model, x)
    col_model = RidgeRegression.fit(x, y |> Nx.new_axis(-1))
    col_pred = RidgeRegression.predict(col_model, x)
    assert model == col_model
    assert pred == col_pred
  end
end
