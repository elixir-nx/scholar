ExUnit.start()

defmodule Datasets do
  def get(key) do
    :persistent_term.get({__MODULE__, key})
  end

  def put(key, x_train, x_test, y_train, y_test) do
    :persistent_term.put(
      {__MODULE__, key},
      {df_to_matrix(x_train), df_to_matrix(x_test), df_to_vector(y_train), df_to_vector(y_test)}
    )
  end

  defp df_to_matrix(df) do
    df
    |> Explorer.DataFrame.names()
    |> Enum.map(&(Explorer.Series.to_tensor(df[&1]) |> Nx.new_axis(-1)))
    |> Nx.concatenate(axis: 1)
  end

  defp df_to_vector(df) do
    case Explorer.DataFrame.names(df) do
      [name] -> Explorer.Series.to_tensor(df[name])
      _several -> df |> df_to_matrix() |> Nx.argmax(axis: 1)
    end
  end
end

# Pima Indians Diabetes Data
data = Explorer.DataFrame.from_csv!("test/data/pima.csv", header: false)

x = Explorer.DataFrame.discard(data, [-1])
y = Explorer.DataFrame.select(data, [-1])

x_train = Explorer.DataFrame.slice(x, 0, 500)
x_test = Explorer.DataFrame.slice(x, 500, 500)

y_train = Explorer.DataFrame.slice(y, 0, 500)
y_test = Explorer.DataFrame.slice(y, 500, 500)

Datasets.put(:pima, x_train, x_test, y_train, y_test)

# IRIS
df = Explorer.Datasets.iris()

train_ids = for n <- 0..149, rem(n, 5) != 0, do: n
test_ids = for n <- 0..149, rem(n, 5) == 0, do: n

x = Explorer.DataFrame.discard(df, ["species"])
y = Explorer.DataFrame.select(df, ["species"]) |> Explorer.DataFrame.dummies(["species"])

x_train = Explorer.DataFrame.slice(x, train_ids)
x_test = Explorer.DataFrame.slice(x, test_ids)

y_train = Explorer.DataFrame.slice(y, train_ids)
y_test = Explorer.DataFrame.slice(y, test_ids)

Datasets.put(:iris, x_train, x_test, y_train, y_test)
