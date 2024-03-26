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
    assert_all_close(expected, predicted, atol: 1.0e-1)
  end
  
  test "toy bayesian ride expanded" do
    x = Nx.tensor([[1, 1], [2, 2], [6, 6], [8, 8], [10, 10]])
    y = Nx.tensor([1, 2, 6, 8, 10])
    clf = BayesianRidgeRegression.fit(x, y)
    test = Nx.tensor([[1, 1], [3, 3], [4, 4]])
    expected = Nx.tensor([1, 3, 4])
    predicted = BayesianRidgeRegression.predict(clf, test)
    IO.inspect(clf)
    IO.inspect(predicted)
    assert false            
  end  
end
