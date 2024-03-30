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

  test "ridge vs bayesian ridge: parameters" do
    x = Nx.tensor([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = Nx.tensor([1, 2, 3, 2, 0, 4, 5])
    brr = BayesianRidgeRegression.fit(x, y)
    rr = RidgeRegression.fit(x, y, alpha: brr.lambda / brr.alpha)
    assert_all_close(brr.coefficients, rr.coefficients, atol: 1.0e-2)
    assert_all_close(brr.intercept, rr.intercept, atol: 1.0e-2)
  end

  test "ridge vs bayesian ridge: weights" do
    x = Nx.tensor([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = Nx.tensor([1, 2, 3, 2, 0, 4, 5])
    w = Nx.tensor([4, 3, 3, 1, 1, 2, 3])
    brr = BayesianRidgeRegression.fit(x, y, sample_weights: w)
    rr = RidgeRegression.fit(x, y, alpha: brr.lambda / brr.alpha, sample_weights: w)
    assert_all_close(brr.coefficients, rr.coefficients, atol: 1.0e-2)
    assert_all_close(brr.intercept, rr.intercept, atol: 1.0e-2)
  end

  test "compute scores" do
    {x, y} = diabetes_data()
    n_samples = 50 - 1
    x = x[[0..n_samples, ..]]
    y = y[[0..n_samples, ..]]
    eps = Nx.Constants.smallest_positive_normal({:f, 64})
    alpha = Nx.divide(1, Nx.add(Nx.variance(x), eps))
    lambda = Nx.tensor(1.0)
    alpha_1 = 0.1
    alpha_2 = 0.1
    lambda_1 = 0.1
    lambda_2 = 0.1
    # compute score
    score = compute_score(x, y, alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2)
    brr =
      BayesianRidgeRegression.fit(x, y,
        alpha_1: alpha_1,
        alpha_2: alpha_2,
        lambda_1: lambda_1,
        lambda_2: lambda_2,
        fit_intercept?: true,
        iterations: 1
      )
    compare_scores = Nx.divide(Nx.subtract(score, brr.score), score)
    check = Nx.less(compare_scores, 0.05) |> Nx.flatten()
    assert check == Nx.tensor([1], type: {:u, 8})
  end

  defnp compute_score(x, y, alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2) do
    {n_samples, _} = Nx.shape(x)
    lambda_score = lambda_1 * Nx.log(lambda) - lambda_2 * lambda
    alpha_score = alpha_1 * Nx.log(alpha) - alpha_2 * alpha
    m = (1.0 / alpha * Nx.eye(n_samples)) + (1.0 / lambda * Nx.dot(x, Nx.transpose(x)))
    m_inv_dot_y = Nx.LinAlg.solve(m, y)
    logdet = m |> Nx.LinAlg.determinant() |> Nx.log()

    y_score =
      -0.5 * (logdet + Nx.dot(Nx.transpose(y), m_inv_dot_y) + n_samples * Nx.log(2 * Nx.Constants.pi()))

    alpha_score + lambda_score + y_score
  end

  test "constant inputs: prediction" do
    assert false
  end

  test "constant inputs: variance" do
    assert false
  end

  test "n_features > n_samples" do
    assert false
  end
end
