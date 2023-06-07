defmodule Scholar.Linear.LogisticRegressionTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.LogisticRegression
  doctest LogisticRegression

  test "Pima Indians Diabetes Data - binary logistic regression test" do
    {x_train, x_test, y_train, y_test} = Datasets.get(:pima) |> Nx.backend_transfer(EXLA.Backend)
    y_train = Nx.squeeze(y_train, axes: [1])
    y_test = Nx.squeeze(y_test, axes: [1])

    model =
      LogisticRegression.fit(x_train, y_train,
        num_classes: 2,
        iterations: 1000
      )

    res = Scholar.Linear.LogisticRegression.predict(model, x_test)
    assert Scholar.Metrics.accuracy(y_test, res) >= 0.6
  end

  test "Iris Data Set - multinomial logistic regression test for multinomial data" do
    {x_train, x_test, y_train, y_test} = Datasets.get(:iris) |> Nx.backend_transfer(EXLA.Backend)
    y_train = Nx.argmax(y_train, axis: 1)
    y_test = Nx.argmax(y_test, axis: 1)

    model = LogisticRegression.fit(x_train, y_train, num_classes: 3)
    res = LogisticRegression.predict(model, x_test)
    assert Scholar.Metrics.accuracy(y_test, res) >= 0.965
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
                   "invalid value for :optimizer option: expected :optimizer to be either a valid 0-arity function in Optimus.Optimizers or a valid {init_fn, update_fn} tuple",
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
                   "expected y to have shape {n_samples}, got tensor with shape: {2, 2}",
                   fn -> LogisticRegression.fit(x, y, num_classes: 2) end
    end
  end
end
