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
    num_folds = opts[:k]
    chunk_size = div(Nx.axis_size(data, 0), num_folds)

    mask =
      Nx.transpose(Nx.tri(num_folds - 1, num_folds))
      |> Nx.new_axis(-1)
      |> Nx.broadcast({num_folds, num_folds - 1, chunk_size})
      |> Nx.reshape({num_folds, :auto})

    train_indices = Nx.iota({(num_folds - 1) * chunk_size}) |> Nx.tile([num_folds, 1])
    train_indices = Nx.select(mask, train_indices + chunk_size, train_indices)
    validation_indices = Nx.iota({num_folds, chunk_size})

    case opts[:shuffle] do
      true ->
        shuffle = Nx.iota({chunk_size * num_folds})
        {shuffle, _} = Nx.Random.shuffle(key, shuffle)
        train_indices = Nx.take(shuffle, train_indices)
        validation_indices = Nx.take(shuffle, validation_indices)
        {train_indices, validation_indices}

      false ->
        {train_indices, validation_indices}
    end
  end
end
