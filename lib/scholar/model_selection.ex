defmodule Scholar.ModelSelection do
  @moduledoc """
  Module containing cross validation, splitting function, and other model selection methods.
  """

  @doc """
  Perform K-Fold split on the given data.

  ## Examples

      iex> x = Nx.iota({7, 2})
      iex> Scholar.ModelSelection.k_fold_split(x, 2) |> Enum.to_list()
      [
        {Nx.tensor(
          [
            [6, 7],
            [8, 9],
            [10, 11]
          ]
        ),
        Nx.tensor(
          [
            [0, 1],
            [2, 3],
            [4, 5]
          ]
        )},
        {Nx.tensor(
          [
            [0, 1],
            [2, 3],
            [4, 5]
          ]
        ),
        Nx.tensor(
          [
            [6, 7],
            [8, 9],
            [10, 11]
          ]
        )}
      ]
  """
  def k_fold_split(x, k) do
    Stream.resource(
      fn ->
        x = Nx.tensor(x)
        fold_size = floor(Nx.axis_size(x, 0) / k)
        slices = for i <- 0..(k - 1), do: (i * fold_size)..((i + 1) * fold_size - 1)
        {slices, 0, k}
      end,
      fn
        {list, k, k} ->
          {:halt, {list, k, k}}

        {list, current, k} ->
          {left, [test | right]} = Enum.split(list, current)

          tensors =
            case {left, right} do
              {[], _} ->
                x[concat_ranges(right)]

              {_, []} ->
                x[concat_ranges(left)]

              {[_ | _], [_ | _]} ->
                Nx.concatenate([x[concat_ranges(left)], x[concat_ranges(right)]])
            end

          {[{tensors, x[test]}], {list, current + 1, k}}
      end,
      fn _ -> :ok end
    )
  end

  # Receive a list of contiguous ranges and returns a range with first first and last last.
  defp concat_ranges([first.._ | _] = list), do: first..last_last(list)

  defp last_last([_..last]), do: last
  defp last_last([_ | tail]), do: last_last(tail)

  @doc """
  General interface of cross validation.

  ## Examples

      iex> folding_fun = fn x -> Scholar.ModelSelection.k_fold_split(x, 3) end
      iex> scoring_fun = fn x, y ->
      ...>   opts = [fit_intercept?: true]
      ...>   {x_train, x_test} = x
      ...>   {y_train, y_test} = y
      ...>   model = Scholar.Linear.LinearRegression.fit(x_train, y_train, opts)
      ...>   y_pred = Scholar.Linear.LinearRegression.predict(model, x_test)
      ...>   mse = Scholar.Metrics.Regression.mean_square_error(y_test, y_pred)
      ...>   mae = Scholar.Metrics.Regression.mean_absolute_error(y_test, y_pred)
      ...>   [mse, mae]
      ...> end
      iex> x = Nx.iota({7, 2})
      iex> y = Nx.tensor([0, 1, 2, 0, 1, 1, 0])
      iex> Scholar.ModelSelection.cross_validate(x, y, folding_fun, scoring_fun)
      #Nx.Tensor<
        f32[2][3]
        [
          [1.5700000524520874, 1.2149654626846313, 0.004999990575015545],
          [1.100000023841858, 1.0735294818878174, 0.04999995231628418]
        ]
      >
  """
  def cross_validate(x, y, folding_fun, scoring_fun) do
    Stream.zip([folding_fun.(x), folding_fun.(y)])
    |> Enum.map(fn {x, y} -> scoring_fun.(x, y) |> Nx.stack() end)
    |> Nx.stack(axis: 1)
  end

  @doc """
  General interface of weighted cross validation.

  ## Examples

      iex> folding_fun = fn x -> Scholar.ModelSelection.k_fold_split(x, 3) end
      iex> scoring_fun = fn x, y, weights ->
      ...>   opts = [fit_intercept?: true]
      ...>   {x_train, x_test} = x
      ...>   {y_train, y_test} = y
      ...>   {weights_train, _weights_test} = weights
      ...>   opts = Keyword.put(opts, :sample_weights, weights_train)
      ...>   model = Scholar.Linear.LinearRegression.fit(x_train, y_train, opts)
      ...>   y_pred = Scholar.Linear.LinearRegression.predict(model, x_test)
      ...>   mse = Scholar.Metrics.Regression.mean_square_error(y_test, y_pred)
      ...>   mae = Scholar.Metrics.Regression.mean_absolute_error(y_test, y_pred)
      ...>   [mse, mae]
      ...> end
      iex> x = Nx.iota({7, 2})
      iex> y = Nx.tensor([0, 1, 2, 0, 1, 1, 0])
      iex> weights = Nx.tensor([1, 2, 1, 2, 1, 2, 1])
      iex> Scholar.ModelSelection.weighted_cross_validate(x, y, weights, folding_fun, scoring_fun)
      #Nx.Tensor<
        f32[2][3]
        [
          [0.5010331869125366, 1.1419668197631836, 0.35123956203460693],
          [0.5227273106575012, 1.0526316165924072, 0.5909090042114258]
        ]
      >

  """
  def weighted_cross_validate(x, y, weights, folding_fun, scoring_fun) do
    Stream.zip([folding_fun.(x), folding_fun.(y), folding_fun.(weights)])
    |> Enum.map(fn {x, y, weights} -> scoring_fun.(x, y, weights) |> Nx.stack() end)
    |> Nx.stack(axis: 1)
  end

  defp combinations([]), do: [[]]

  defp combinations([{name, values} | opts]) do
    for subcombination <- combinations(opts), value <- values do
      [{name, value} | subcombination]
    end
  end

  @doc """
  General interface of grid search.
  If you want to use `opts` in some functions inside `scoring_fun`, you need to pass it as a parameter
  like in the example below.

  ## Examples

      iex> folding_fun = fn x -> Scholar.ModelSelection.k_fold_split(x, 3) end
      iex> scoring_fun = fn x, y, opts ->
      ...>   {x_train, x_test} = x
      ...>   {y_train, y_test} = y
      ...>   model = Scholar.Linear.LogisticRegression.fit(x_train, y_train, opts)
      ...>   y_pred = Scholar.Linear.LogisticRegression.predict(model, x_test)
      ...>   mse = Scholar.Metrics.Regression.mean_square_error(y_test, y_pred)
      ...>   mae = Scholar.Metrics.Regression.mean_absolute_error(y_test, y_pred)
      ...>   [mse, mae]
      ...> end
      iex> x = Nx.iota({7, 2})
      iex> y = Nx.tensor([0, 1, 2, 0, 1, 1, 0])
      iex> opts = [
      ...>   num_classes: [3],
      ...>   iterations: [10, 20, 50],
      ...>   optimizer: [Polaris.Optimizers.adam(learning_rate: 0.005), Polaris.Optimizers.adam(learning_rate: 0.01)],
      ...> ]
      iex> Scholar.ModelSelection.grid_search(x, y, folding_fun, scoring_fun, opts)
  """
  def grid_search(x, y, folding_fun, scoring_fun, opts) do
    params = combinations(opts)

    for param <- params do
      scoring_function = check_arity(scoring_fun, params)

      [
        hyperparameters: param,
        score: Nx.mean(cross_validate(x, y, folding_fun, scoring_function), axes: [1])
      ]
    end
  end

  @doc """
  General interface of weighted grid search.

  If you want to use `opts` in some functions inside `scoring_fun`, you need to pass it as a parameter
  like in the example below.

  ## Examples

      iex> folding_fun = fn x -> Scholar.ModelSelection.k_fold_split(x, 3) end
      iex> scoring_fun = fn x, y, weights, opts ->
      ...>   {x_train, x_test} = x
      ...>   {y_train, y_test} = y
      ...>   {weights_train, _weights_test} = weights
      ...>   opts = Keyword.put(opts, :sample_weights, weights_train)
      ...>   model = Scholar.Linear.RidgeRegression.fit(x_train, y_train, opts)
      ...>   y_pred = Scholar.Linear.RidgeRegression.predict(model, x_test)
      ...>   mse = Scholar.Metrics.Regression.mean_square_error(y_test, y_pred)
      ...>   mae = Scholar.Metrics.Regression.mean_absolute_error(y_test, y_pred)
      ...>   [mse, mae]
      ...> end
      iex> x = Nx.iota({7, 2})
      iex> y = Nx.tensor([0, 1, 2, 0, 1, 1, 0])
      iex> weights = [Nx.tensor([1, 2, 1, 2, 1, 2, 1]), Nx.tensor([2, 1, 2, 1, 2, 1, 2])]
      iex> opts = [
      ...>   alpha: [0, 1, 5],
      ...>   fit_intercept?: [true, false],
      ...> ]
      iex> Scholar.ModelSelection.weighted_grid_search(x, y, weights, folding_fun, scoring_fun, opts)
  """
  def weighted_grid_search(x, y, weights, folding_fun, scoring_fun, opts) do
    params = combinations(opts)

    for weight <- weights,
        param <- params do
      scoring_function = check_arity_weighted(scoring_fun, params)

      [
        weights: weight,
        hyperparameters: param,
        score:
          Nx.mean(weighted_cross_validate(x, y, weight, folding_fun, scoring_function),
            axes: [1]
          )
      ]
    end
  end

  defp check_arity(scoring_fun, opts) do
    cond do
      is_function(scoring_fun, 3) ->
        fn x, y -> scoring_fun.(x, y, opts) end

      is_function(scoring_fun, 2) and opts != [] ->
        raise ArgumentError, "no options must be given if scoring_fun expects two arguments"

      is_function(scoring_fun, 2) ->
        scoring_fun

      true ->
        raise ArgumentError,
              "expected scoring_fun to be a function of arity 2 or 3, got: #{inspect(scoring_fun)}"
    end
  end

  defp check_arity_weighted(scoring_fun, opts) do
    cond do
      is_function(scoring_fun, 4) ->
        fn x, y, weights -> scoring_fun.(x, y, weights, opts) end

      is_function(scoring_fun, 3) and opts != [] ->
        raise ArgumentError, "no options must be given if scoring_fun expects three arguments"

      is_function(scoring_fun, 3) ->
        scoring_fun

      true ->
        raise ArgumentError,
              "expected scoring_fun to be a function of arity 3 or 4, got: #{inspect(scoring_fun)}"
    end
  end
end
