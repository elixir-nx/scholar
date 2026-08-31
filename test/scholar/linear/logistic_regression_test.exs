defmodule Scholar.Linear.LogisticRegressionTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.LogisticRegression
  doctest LogisticRegression

  test "multinomial logistic regression" do
    x_train =
      Nx.tensor([
        [-2.0, -2.0],
        [-2.0, -1.0],
        [-1.0, -2.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, -2.0],
        [2.0, -1.0],
        [1.0, -2.0]
      ])

    y_train = Nx.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    x_test = Nx.tensor([[-1.5, -1.5], [1.5, 1.5], [1.5, -1.5]])
    y_test = Nx.tensor([0, 1, 2])

    model =
      LogisticRegression.fit(x_train, y_train,
        num_classes: 3,
        alpha: 0.0,
        max_iterations: 10
      )

    res = LogisticRegression.predict(model, x_test)
    accuracy = Scholar.Metrics.Classification.accuracy(res, y_test)

    assert_all_close(accuracy, Nx.tensor(1.0))
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
      x = Nx.tensor([[-2.0], [-1.0], [1.0], [2.0]])
      y = Nx.tensor([0, 0, 1, 1])
      model = LogisticRegression.fit(x, y, num_classes: 2, max_iterations: 10)
      y_pred = LogisticRegression.predict(model, x)
      accuracy = Scholar.Metrics.Classification.accuracy(y, y_pred)
      assert_all_close(accuracy, Nx.tensor(1.0))
    end

    test "2D" do
      x = Nx.tensor([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
      y = Nx.tensor([0, 0, 1, 1])
      model = LogisticRegression.fit(x, y, num_classes: 2, max_iterations: 10)
      y_pred = LogisticRegression.predict(model, x)
      accuracy = Scholar.Metrics.Classification.accuracy(y, y_pred)
      assert_all_close(accuracy, Nx.tensor(1.0))
    end
  end
end
