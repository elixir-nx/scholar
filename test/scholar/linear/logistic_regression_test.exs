defmodule Scholar.Linear.LogisticRegressionTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.LogisticRegression
  doctest LogisticRegression

  test "Iris Data Set - multinomial logistic regression test" do
    {x_train, x_test, y_train, y_test} = iris_data()

    model = LogisticRegression.fit(x_train, y_train, num_classes: 3)
    res = LogisticRegression.predict(model, x_test)
    accuracy = Scholar.Metrics.Classification.accuracy(res, y_test)

    assert Nx.greater_equal(accuracy, 0.96) == Nx.u8(1)
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

    test "when :optimizer is invalid" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([1, 2])

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :optimizer option: expected :optimizer to be either a valid 0-arity function in Polaris.Optimizers or a valid {init_fn, update_fn} tuple",
                   fn ->
                     LogisticRegression.fit(x, y,
                       num_classes: 2,
                       optimizer: :invalid_optimizer
                     )
                   end
    end

    test "when :iterations is not a positive integer" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([1, 2])

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :iterations option: expected positive integer, got: 0",
                   fn ->
                     LogisticRegression.fit(x, y, num_classes: 2, iterations: 0)
                   end
    end

    test "when training vector size is invalid" do
      x = Nx.tensor([5, 6])
      y = Nx.tensor([1, 2])

      assert_raise ArgumentError,
                   "expected x to have shape {n_samples, n_features}, got tensor with shape: {2}",
                   fn -> LogisticRegression.fit(x, y, num_classes: 2) end
    end

    test "when target vector size is invalid" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([[0, 1], [1, 0]])

      assert_raise ArgumentError,
                   "Scholar.Linear.LogisticRegression expected y to have shape {n_samples}, got tensor with shape: {2, 2}",
                   fn -> LogisticRegression.fit(x, y, num_classes: 2) end
    end
  end

  describe "column target tests" do
    @tag :wip
    test "column target" do
      {x_train, _, y_train, _} = iris_data()

      model = LogisticRegression.fit(x_train, y_train, num_classes: 3)
      pred = LogisticRegression.predict(model, x_train)
      col_model = LogisticRegression.fit(x_train, y_train |> Nx.new_axis(-1), num_classes: 3)
      col_pred = LogisticRegression.predict(col_model, x_train)
      assert model == col_model
      assert pred == col_pred
    end
  end
end
