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
                Nx.concatenate(Enum.map(right, fn range -> x[range] end), axis: 0)

              {_, []} ->
                Nx.concatenate(Enum.map(left, fn range -> x[range] end), axis: 0)

              {[_ | _], [_ | _]} ->
                Nx.concatenate([
                  Nx.concatenate(Enum.map(left, fn range -> x[range] end), axis: 0),
                  Nx.concatenate(Enum.map(right, fn range -> x[range] end), axis: 0)
                ])
            end

          {[{tensors, x[test]}], {list, current + 1, k}}
      end,
      fn _ -> :ok end
    )
  end

  @doc """
  General interface of cross validation.

  ## Examples

      iex> folding_fun = fn x -> Scholar.ModelSelection.k_fold_split(x, 3) end
      iex> scoring_fun = fn x, y ->
      ...>   {x_train, x_test} = x
      ...>   {y_train, y_test} = y
      ...>   model = Scholar.Linear.LinearRegression.fit(x_train, y_train)
      ...>   y_pred = Scholar.Linear.LinearRegression.predict(model, x_test)
      ...>   mse = Scholar.Metrics.mean_square_error(y_test, y_pred)
      ...>   mae = Scholar.Metrics.mean_absolute_error(y_test, y_pred)
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
end
