defmodule Scholar.Linear.BayesianRidgeRegressionTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.BayesianRidgeRegression
  doctest BayesianRidgeRegression

  test "toy bayesian ridge" do
    x = Nx.tensor([[1], [2], [6], [8], [10]])
    y = Nx.tensor([1, 2, 6, 8, 10])
    clf = BayesianRidgeRegression.fit(x, y)
    test = Nx.tensor([[1], [3], [4]])
    expected = Nx.tensor([1, 3, 4])
    predicted = BayesianRidgeRegression.predict(clf, test)
    assert_all_close(expected, predicted, atol: 1.0e-6)
  end
  
  test "multi column toy bayesian ridge" do
    x = Nx.tensor([
      [1, 5], [2, 6], [6, 6], [8, 4], [10, 0],
      [5, 5], [6, 2], [6, 4], [4, 2], [0, 10],      
    ])
    true_coef = Nx.tensor([0.5, 0.5])
    y = Nx.dot(x, true_coef)
    clf = BayesianRidgeRegression.fit(x, y)
    test = Nx.tensor([[1, 1], [3, 3], [4, 4]])
    expected = Nx.tensor([1, 3, 4])
    predicted = BayesianRidgeRegression.predict(clf, test)
    assert_all_close(expected, predicted, atol: 1.0e-3)
  end

  test "compare ridge vs bayesian ridge" do
    # go to sklearn tests and copy
    assert false
  end

end
