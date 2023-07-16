defmodule Scholar.ModelSelection.KFold do
  @moduledoc """
  K-Fold Cross Validation
  """

  import Nx.Defn
  import Scholar.Shared

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
    data = data[[0..(div(Nx.axis_size(data, 0), k) * k - 1)]]
    data = Nx.reshape(data, get_shape(data, k: opts[:k]))

    data =
      case opts[:shuffle] do
        true ->
          {shuffled, _} = Nx.Random.shuffle(key, data)
          shuffled

        false ->
          data
      end

    train_indices = Nx.iota({k - 1}) |> Nx.tile([k, 1])

    train_indices = train_indices + Nx.transpose(Nx.tri(k - 1, k))
    validation_indices = Nx.iota({k})
    %__MODULE__{data: data, train_indices: train_indices, validation_indices: validation_indices}
  end

  deftransform get_shape(data, opts \\ []) do
    [_ | rest_shape] = Tuple.to_list(Nx.shape(data))
    List.to_tuple([opts[:k], div(Nx.axis_size(data, 0), opts[:k])] ++ rest_shape)
  end
end
