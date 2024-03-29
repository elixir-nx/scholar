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

  @tag :wip
  test "compute scores" do
    {x, y, noise} = data()

    eps = Nx.Constants.smallest_positive_normal({:f, 64})
    alpha = Nx.divide(1, Nx.add(Nx.variance(x), eps))
    lambda = Nx.tensor(1.0)
    alpha_1 = 0.1
    alpha_2 = 0.1
    lambda_1 = 0.1
    lambda_2 = 0.1
    # compute score
    score = compute_score(x, y, alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2)
    IO.inspect(score)
    brr = BayesianRidgeRegression.fit(x, y,
      alpha_1: alpha_1, alpha_2: alpha_2, lambda_1: lambda_1, lambda_2: lambda_2,
      fit_intercept?: false, iterations: 1)
    assert false
  end

  defnp compute_score(x, y, alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2) do
    {n_samples, _} = Nx.shape(x)
    lambda_score = lambda_1 + Nx.log(lambda) - lambda_2 * lambda
    alpha_score =  alpha_1 + Nx.log(alpha_2) - alpha_2 * alpha
    m =  (1.0 / alpha) * Nx.eye(n_samples) + (1.0 / lambda) * Nx.dot(x, Nx.transpose(x))
    m_inv_dot_y = Nx.LinAlg.solve(m, y)
    logdet = m |> Nx.LinAlg.determinant |> Nx.log()
    y_score = -0.5 * (logdet + Nx.dot(y, m_inv_dot_y) +
                        n_samples * Nx.log(2 * Nx.Constants.pi()))
    score_alpha + score_lambda + y_score
  end

  # adapted from the linear regression notebook
  # https://hexdocs.pm/scholar/linear_regression.html
  defnp data do
    key = Nx.Random.key(42)
    size = 30
    {x, new_key} = Nx.Random.normal(key, 0, 2, shape: {size, 3}, type: :f64)
    {noise, _} = Nx.Random.normal(new_key, 0, 1, shape: {size}, type: :f64)
    w = Nx.tensor([1, 2, 3])
    y = Nx.dot(x, w) + 4
    {x, y, noise}
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
