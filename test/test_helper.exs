ExUnit.start(exclude: [:slow])

alias Explorer.DataFrame, as: DF

defmodule Datasets do
  def get(key) do
    :persistent_term.get({__MODULE__, key})
  end

  def put(key, x_train, x_test, y_train, y_test) do
    :persistent_term.put(
      {__MODULE__, key},
      {Nx.stack(x_train, axis: 1), Nx.stack(x_test, axis: 1), Nx.stack(y_train, axis: 1),
       Nx.stack(y_test, axis: 1)}
    )
  end
end

# Pima Indians Diabetes Data
data = DF.from_csv!("test/data/pima.csv", header: false)

x = DF.discard(data, [-1])
y = DF.select(data, [-1])

x_train = DF.slice(x, 0, 600)
x_test = DF.slice(x, 600, 500)

y_train = DF.slice(y, 0, 600)
y_test = DF.slice(y, 600, 500)

Datasets.put(:pima, x_train, x_test, y_train, y_test)

# IRIS
df = Explorer.Datasets.iris()

train_ids = for n <- 0..149, rem(n, 5) != 0, do: n
test_ids = for n <- 0..149, rem(n, 5) == 0, do: n

x = DF.discard(df, ["species"])
y = DF.select(df, ["species"]) |> DF.dummies(["species"])

x_train = DF.slice(x, train_ids)
x_test = DF.slice(x, test_ids)

y_train = DF.slice(y, train_ids)
y_test = DF.slice(y, test_ids)

Datasets.put(:iris, x_train, x_test, y_train, y_test)
