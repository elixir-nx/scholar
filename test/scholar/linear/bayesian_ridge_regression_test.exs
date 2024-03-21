defmodule Scholar.Linear.BayesianRidgeRegressionTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.BayesianRidgeRegression
  doctest BayesianRidgeRegression

  test "toy bayesian ridge" do
    x = Nx.tensor([[1], [2], [6], [8], [10]])
    {u, s, vh} = Nx.LinAlg.svd(x, full_matrices?: false)
    eigen_vals = Nx.pow(s, 2)
    y = Nx.tensor([1, 2, 6, 8, 10])
    alpha = 1
    lambda = 1
    xt_y = Nx.dot(Nx.transpose(x), y)
    IO.inspect(xt_y)
    regularization = # vh / (eigen_vals + lambda / alpha)
      Nx.divide(vh, Nx.divide(Nx.add(eigen_vals, lambda), alpha))
    IO.inspect(regularization)
    reg_transpose = Nx.dot(regularization, xt_y)
    coef = Nx.dot(Nx.transpose(vh), reg_transpose)
    IO.inspect(coef)
    
    
    clf = BayesianRidgeRegression.fit(x, y)
    test = Nx.tensor([[1], [3], [4]])
    expected_predict = Nx.tensor([1, 3, 4])

    assert false
  end
  
end
