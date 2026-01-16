defmodule Scholar.Linear.LogisticRegressionTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.LogisticRegression
  doctest LogisticRegression

  test "Iris Data Set - multinomial logistic regression test" do
    {x_train, x_test, y_train, y_test} = iris_data()

    model = LogisticRegression.fit(x_train, y_train, num_classes: 3, alpha: 0.0)
    res = LogisticRegression.predict(model, x_test)
    accuracy = Scholar.Metrics.Classification.accuracy(res, y_test)

    assert Nx.to_number(accuracy) >= 0.96
  end

  describe "errors" do
    test "when :num_classes is invalid" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([1, 2])

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :num_classes option: expected positive integer, got: -3",
                   fn ->
                     LogisticRegression.fit(x, y, num_classes: -3)
                   end

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :num_classes option: expected positive integer, got: 2.0",
                   fn ->
                     LogisticRegression.fit(x, y, num_classes: 2.0)
                   end
    end

    test "when missing :num_classes option" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([0, 1])

      assert_raise NimbleOptions.ValidationError,
                   "required :num_classes option not found, received options: []",
                   fn -> LogisticRegression.fit(x, y) end
    end

    test "when :max_iterations is not a positive integer" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([1, 2])

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :max_iterations option: expected positive integer, got: 0",
                   fn ->
                     LogisticRegression.fit(x, y, num_classes: 2, max_iterations: 0)
                   end
    end

    test "when training vector size is invalid" do
      x = Nx.tensor([5, 6])
      y = Nx.tensor([1, 2])

      assert_raise ArgumentError,
                   "expected x to have shape {num_samples, num_features}, got tensor with shape: {2}",
                   fn -> LogisticRegression.fit(x, y, num_classes: 2) end
    end

    test "when target vector size is invalid" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([[0, 1], [1, 0]])

      assert_raise ArgumentError,
                   """
                   expected y to have shape {num_samples}, \
                   got tensor with shape: {2, 2}\
                   """,
                   fn -> LogisticRegression.fit(x, y, num_classes: 2) end
    end
  end

  describe "linearly separable data" do
    test "1D" do
      key = Nx.Random.key(12)
      {x1, key} = Nx.Random.uniform(key, -2, -1, shape: {1000, 1})
      {x2, _key} = Nx.Random.uniform(key, 1, 2, shape: {1000, 1})
      x = Nx.concatenate([x1, x2])
      y1 = Nx.broadcast(0, {1000})
      y2 = Nx.broadcast(1, {1000})
      y = Nx.concatenate([y1, y2])
      model = LogisticRegression.fit(x, y, num_classes: 2)
      y_pred = LogisticRegression.predict(model, x)
      accuracy = Scholar.Metrics.Classification.accuracy(y, y_pred)
      assert Nx.to_number(accuracy) == 1.0
    end

    test "2D" do
      key = Nx.Random.key(12)
      {x1, key} = Nx.Random.uniform(key, -2, -1, shape: {1000, 2})
      {x2, _key} = Nx.Random.uniform(key, 1, 2, shape: {1000, 2})
      x = Nx.concatenate([x1, x2])
      y1 = Nx.broadcast(0, {1000})
      y2 = Nx.broadcast(1, {1000})
      y = Nx.concatenate([y1, y2])
      model = LogisticRegression.fit(x, y, num_classes: 2)
      y_pred = LogisticRegression.predict(model, x)
      accuracy = Scholar.Metrics.Classification.accuracy(y, y_pred)
      assert Nx.to_number(accuracy) == 1.0
    end
  end
end
