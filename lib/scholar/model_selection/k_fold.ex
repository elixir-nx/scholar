defmodule Scholar.ModelSelection.KFold do
  @moduledoc """
  K-Fold Cross Validation
  """

  @doc """
  Perform K-Fold split on the given data.

  ## Examples

      iex> x = Nx.iota({7, 2})
      iex> y = Nx.iota({7})
      iex> Scholar.ModelSelection.KFold.cross_fold(x, y, 2) |> Enum.to_list()
      [
        {{[
            Nx.tensor(
              [
                [6, 7],
                [8, 9],
                [10, 11]
              ]
            )
          ],
          Nx.tensor(
            [
              [0, 1],
              [2, 3],
              [4, 5]
            ]
          )},
        {[
            Nx.tensor(
              [3, 4, 5]
            )
          ],
          Nx.tensor(
            [0, 1, 2]
          )}},
        {{[
            Nx.tensor(
              [
                [0, 1],
                [2, 3],
                [4, 5]
              ]
            )
          ],
          Nx.tensor(
            [
              [6, 7],
              [8, 9],
              [10, 11]
            ]
          )},
        {[
            Nx.tensor(
              [0, 1, 2]
            )
          ],
          Nx.tensor(
            [3, 4, 5]
          )}}
      ]
  """
  def cross_fold(x, y, k) do
    Stream.zip([k_fold_split(x, k), k_fold_split(y, k)])
  end

  defp k_fold_split(x, k) do
    Stream.resource(
      fn ->
        fold_size = floor(Nx.axis_size(x, 0) / k)
        {x |> Nx.to_batched(fold_size, leftover: :discard) |> Enum.to_list(), 0, k}
      end,
      fn
        {list, k, k} ->
          {:halt, {list, k, k}}

        {list, current, k} ->
          {left, [test | right]} = Enum.split(list, current)
          {[{left ++ right, test}], {list, current + 1, k}}
      end,
      fn _ -> :ok end
    )
  end

  @doc """
  Perform K-Fold cross validation on the given data.

  Except data, labels, and number of folds, the function also takes a function
  that takes a tuple of training and testing data, calculates model and predictions
  and returns a list of metrics based on predictions.


  ## Examples

      iex> inner_loop = fn {{x_train, x_test}, {y_train, y_test}} ->
      ...>   x_train = Nx.concatenate(x_train)
      ...>   y_train = Nx.concatenate(y_train)
      ...>   model = Scholar.Linear.LinearRegression.fit(x_train, y_train)
      ...>   y_pred = Scholar.Linear.LinearRegression.predict(model, x_test)
      ...>   mse = Scholar.Metrics.mean_square_error(y_test, y_pred)
      ...>   mae = Scholar.Metrics.mean_absolute_error(y_test, y_pred)
      ...>   [mse, mae]
      ...> end
      iex> x = Nx.iota({7, 2})
      iex> y = Nx.tensor([0, 1, 2, 0, 1, 1, 0])
      iex> Scholar.ModelSelection.KFold.cross_validation(x, y, 3, inner_loop)
      #Nx.Tensor<
        f32[2][3]
        [
          [1.5700000524520874, 1.2149654626846313, 0.004999990575015545],
          [1.100000023841858, 1.0735294818878174, 0.04999995231628418]
        ]
      >
  """
  def cross_validation(x, y, k, fun) do
    cross_fold(x, y, k)
    |> Enum.map(fun)
    |> Enum.map(&Nx.stack/1)
    |> Nx.stack(axis: 1)
  end
end
