defmodule Scholar.Linear.LinearRegressionTest do
  use ExUnit.Case, async: true

  describe "Logistic Regression" do
    def df_to_tensor(df) do
      df
      |> Explorer.DataFrame.names()
      |> Enum.map(&(Explorer.Series.to_tensor(df[&1]) |> Nx.new_axis(-1)))
      |> Nx.concatenate(axis: 1)
    end

    test "Pima Indians Diabetes Data - binary logistic regression test" do
      {:ok, data} =
        Explorer.DataFrame.from_csv(Path.join(__DIR__, "test_data/pima.csv"), header: false)

      x =
        Explorer.DataFrame.select(data, &String.ends_with?(&1, "column_9"), :drop)
        |> df_to_tensor()

      y = Explorer.DataFrame.select(data, &String.ends_with?(&1, "column_9")) |> df_to_tensor()
      x_train = x[[0..500//1, 0..-1//1]]
      x_test = x[[501..-1//1, 0..-1//1]]

      y_train = y[[0..500//1]] |> Nx.squeeze()
      y_test = y[[501..-1//1]] |> Nx.squeeze()

      model =
        Scholar.Linear.LogisticRegression.fit(x_train, y_train,
          num_classes: 2,
          iterations: 1000,
          learning_rate: 0.0085
        )

      res = Scholar.Linear.LogisticRegression.predict(model, x_test)
      assert Scholar.Metrics.accuracy(y_test, res) >= 0.66
    end

    test "Pima Indians Diabetes Data - multinomial logistic regression test for binary data" do
      {:ok, data} =
        Explorer.DataFrame.from_csv(Path.join(__DIR__, "test_data/pima.csv"), header: false)

      x =
        Explorer.DataFrame.select(data, &String.ends_with?(&1, "column_9"), :drop)
        |> df_to_tensor()

      y = Explorer.DataFrame.select(data, &String.ends_with?(&1, "column_9")) |> df_to_tensor()
      x_train = x[[0..500//1, 0..-1//1]]
      x_test = x[[501..-1//1, 0..-1//1]]

      y_train = y[[0..500//1]] |> Nx.squeeze()
      y_test = y[[501..-1//1]] |> Nx.squeeze()

      model =
        Scholar.Linear.LogisticRegression.fit(x_train, y_train,
          num_classes: 3,
          iterations: 1000,
          learning_rate: 0.0085
        )

      res = Scholar.Linear.LogisticRegression.predict(model, x_test)
      assert Scholar.Metrics.accuracy(y_test, res) >= 0.66
    end

    test "Iris Data Set - multinomial logistic regression test for multinomial data" do
      df = Explorer.Datasets.iris()
      train_ids = for n <- 0..149, rem(n, 5) != 0, do: n
      test_ids = for n <- 0..149, rem(n, 5) == 0, do: n
      train_df = Explorer.DataFrame.take(df, train_ids)
      test_df = Explorer.DataFrame.take(df, test_ids)

      x_train =
        Explorer.DataFrame.select(train_df, &String.ends_with?(&1, "species"), :drop)
        |> df_to_tensor()

      y_train =
        Explorer.DataFrame.select(train_df, &String.ends_with?(&1, "species"))
        |> Explorer.DataFrame.dummies(["species"])
        |> df_to_tensor()
        |> Nx.argmax(axis: 1)

      x_test =
        Explorer.DataFrame.select(test_df, &String.ends_with?(&1, "species"), :drop)
        |> df_to_tensor()

      y_test =
        Explorer.DataFrame.select(test_df, &String.ends_with?(&1, "species"))
        |> Explorer.DataFrame.dummies(["species"])
        |> df_to_tensor()
        |> Nx.argmax(axis: 1)

      model = Scholar.Linear.LogisticRegression.fit(x_train, y_train, num_classes: 3)
      res = Scholar.Linear.LogisticRegression.predict(model, x_test)
      assert Scholar.Metrics.accuracy(y_test, res) >= 0.965
    end

    test "raises on invalid :num_classes" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([1, 2])

      assert_raise ArgumentError, "expected :num_classes to be a positive integer, got: -3", fn ->
        Scholar.Linear.LogisticRegression.fit(x, y, num_classes: -3)
      end

      assert_raise ArgumentError,
                   "expected :num_classes to be a positive integer, got: 2.0",
                   fn ->
                     Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2.0)
                   end
    end

    test "Learning rate is not a positive number" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([1, 2])

      assert_raise ArgumentError,
                   "expected :learning_rate to be a positive number, got: -0.001",
                   fn ->
                     Scholar.Linear.LogisticRegression.fit(x, y,
                       num_classes: 2,
                       learning_rate: -0.001
                     )
                   end
    end

    test "Number of iterations is not a positive integer" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([1, 2])

      assert_raise ArgumentError,
                   "expected :iterations to be a positive integer, got: 0",
                   fn ->
                     Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2, iterations: 0)
                   end
    end

    test "Wrong training vector size" do
      x = Nx.tensor([5, 6])
      y = Nx.tensor([1, 2])

      assert_raise ArgumentError,
                   "expected x to have shape {n_samples, n_features}, got tensor with shape: {2}",
                   fn -> Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2) end
    end

    test "Wrong target vector size" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([[0, 1], [1, 0]])

      assert_raise ArgumentError,
                   "expected y to have shape {n_samples}, got tensor with shape: {2, 2}",
                   fn -> Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2) end
    end

    test "Missing the :num_classes option" do
      x = Nx.tensor([[1, 2], [3, 4]])
      y = Nx.tensor([0, 1])

      assert_raise ArgumentError,
                   "missing option :num_classes",
                   fn -> Scholar.Linear.LogisticRegression.fit(x, y) end
    end
  end
end
