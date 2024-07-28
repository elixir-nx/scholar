defmodule Scholar.Linear.BayesianRidgeRegressionTest do
  import Nx.Defn
  use Scholar.Case, async: true
  alias Scholar.Linear.BayesianRidgeRegression
  alias Scholar.Linear.RidgeRegression
  doctest BayesianRidgeRegression

  test "toy bayesian ridge" do
    x = Nx.tensor([[1], [2], [6], [8], [10]])
    y = Nx.tensor([1, 2, 6, 8, 10])
    clf = BayesianRidgeRegression.fit(x, y)
    test = Nx.tensor([[1], [3], [4]])
    expected = Nx.tensor([1, 3, 4])
    predicted = BayesianRidgeRegression.predict(clf, test)
    assert_all_close(expected, predicted, atol: 1.0e-1)
  end

  test "toy bayesian ridge with column target" do
    x = Nx.tensor([[1], [2], [6], [8], [10]])
    y = Nx.tensor([1, 2, 6, 8, 10])
    model = BayesianRidgeRegression.fit(x, y)
    pred = BayesianRidgeRegression.predict(model, x)
    col_model = BayesianRidgeRegression.fit(x, y |> Nx.new_axis(-1))
    col_pred = BayesianRidgeRegression.predict(col_model, x)
    assert model == col_model
    assert pred == col_pred
  end

  test "2 column target raises" do
    x = Nx.tensor([[1], [2], [6], [8], [10]])
    y = Nx.tensor([1, 2, 6, 8, 10])
    y = Nx.new_axis(y, -1)
    y = Nx.concatenate([y, y], axis: 1)

    message =
      "Scholar.Linear.BayesianRidgeRegression expected y to have shape {n_samples}, got tensor with shape: #{inspect(Nx.shape(y))}"

    assert_raise ArgumentError,
                 message,
                 fn ->
                   BayesianRidgeRegression.fit(x, y)
                 end
  end

  test "ridge vs bayesian ridge: parameters" do
    x = Nx.tensor([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = Nx.tensor([1, 2, 3, 2, 0, 4, 5])
    brr = BayesianRidgeRegression.fit(x, y)
    rr = RidgeRegression.fit(x, y, alpha: Nx.to_number(brr.lambda) / Nx.to_number(brr.alpha))
    assert_all_close(brr.coefficients, rr.coefficients, atol: 1.0e-2)
    assert_all_close(brr.intercept, rr.intercept, atol: 1.0e-2)
  end

  test "ridge vs bayesian ridge: weights" do
    x = Nx.tensor([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = Nx.tensor([1, 2, 3, 2, 0, 4, 5])
    w = Nx.tensor([4, 3, 3, 1, 1, 2, 3])
    brr = BayesianRidgeRegression.fit(x, y, sample_weights: w)

    rr =
      RidgeRegression.fit(x, y,
        alpha: Nx.to_number(brr.lambda) / Nx.to_number(brr.alpha),
        sample_weights: w
      )

    assert_all_close(brr.coefficients, rr.coefficients, atol: 1.0e-2)
    assert_all_close(brr.intercept, rr.intercept, atol: 1.0e-2)
  end

  test "compute scores" do
    {x, y} = diabetes_data()
    eps = Nx.Constants.smallest_positive_normal(:f64)
    alpha = Nx.divide(1, Nx.add(Nx.variance(x), eps))
    lambda = 1.0
    alpha_1 = 0.1
    alpha_2 = 0.1
    lambda_1 = 0.1
    lambda_2 = 0.1
    # compute score
    score = compute_score(x, y, alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2)

    brr =
      BayesianRidgeRegression.fit(x, Nx.flatten(y),
        alpha_1: alpha_1,
        alpha_2: alpha_2,
        lambda_1: lambda_1,
        lambda_2: lambda_2,
        fit_intercept?: true,
        compute_scores?: true,
        iterations: 1
      )

    first_score = brr.scores[0]
    assert_all_close(score, first_score, rtol: 0.05)
  end

  defnp compute_score(x, y, alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2) do
    {n_samples, _} = Nx.shape(x)
    lambda_score = lambda_1 * Nx.log(lambda) - lambda_2 * lambda
    alpha_score = alpha_1 * Nx.log(alpha) - alpha_2 * alpha
    m = 1.0 / alpha * Nx.eye(n_samples) + 1.0 / lambda * Nx.dot(x, [-1], x, [-1])
    m_inv_dot_y = Nx.LinAlg.solve(m, y)
    logdet = m |> Nx.LinAlg.determinant() |> Nx.log()

    y_score =
      -0.5 *
        (logdet + Nx.dot(y, [0], m_inv_dot_y, [0]) + n_samples * Nx.log(2 * Nx.Constants.pi()))

    alpha_score + lambda_score + y_score
  end

  test "constant inputs: prediction. n_features > n_samples" do
    key = Nx.Random.key(42)
    n_samples = 4
    n_features = 5
    {constant_value, new_key} = Nx.Random.uniform(key)
    {x, _} = Nx.Random.uniform(new_key, shape: {n_samples, n_features}, type: :f64)
    y = Nx.broadcast(constant_value, {n_samples})
    expected = Nx.broadcast(constant_value, {n_samples})
    brr = BayesianRidgeRegression.fit(x, y)
    predicted = BayesianRidgeRegression.predict(brr, x)
    assert_all_close(expected, predicted, atol: 0.01)
  end

  test "constant inputs: variance is constant" do
    key = Nx.Random.key(42)
    n_samples = 15
    n_features = 10
    {constant_value, new_key} = Nx.Random.uniform(key)
    {x, _} = Nx.Random.uniform(new_key, shape: {n_samples, n_features}, type: :f64)
    y = Nx.broadcast(constant_value, {n_samples})
    brr = BayesianRidgeRegression.fit(x, y)
    check = Nx.less_equal(brr.sigma, 0.01)
    assert Nx.all(check) == Nx.u8(1)
  end
end
