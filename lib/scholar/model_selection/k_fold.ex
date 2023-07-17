defmodule Scholar.ModelSelection.KFold do
  @moduledoc """
  K-Fold Cross Validation
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:data, :train_indices, :validation_indices]}
  defstruct [:data, :train_indices, :validation_indices]

  opts = [
    k: [
      type: :integer,
      default: 5,
      doc: "Number of folds"
    ],
    shuffle: [
      type: :boolean,
      default: true,
      doc: "Determines if data is shuffled before splitting"
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Used in shuffling data.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Perform K-Fold Cross Validation

  ## Options

  #{NimbleOptions.docs(@opts_schema)}
  """

  deftransform k_fold(data, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    k_fold_n(data, key, opts)
  end

  defnp k_fold_n(data, key, opts \\ []) do
    k = opts[:k]

    mask =
      Nx.transpose(Nx.tri(k - 1, k))
      |> Nx.new_axis(-1)
      |> Nx.broadcast({k, k - 1, div(Nx.axis_size(data, 0), k)})
      |> Nx.reshape({k, :auto})

    train_indices = Nx.iota({(k - 1) * div(Nx.axis_size(data, 0), k)}) |> Nx.tile([k, 1])
    train_indices = Nx.select(mask, train_indices + div(Nx.axis_size(data, 0), k), train_indices)
    validation_indices = Nx.iota({k, div(Nx.axis_size(data, 0), k)})
    case opts[:shuffle] do
      true ->
        shuffle = Nx.iota({div(Nx.axis_size(data, 0), k) * k})
        {shuffle, _} = Nx.Random.shuffle(key, shuffle)
        train_indices = Nx.take(shuffle, train_indices)
        validation_indices = Nx.take(shuffle, validation_indices)
        {train_indices, validation_indices}
      false ->
        {train_indices, validation_indices}
    end
  end
end
