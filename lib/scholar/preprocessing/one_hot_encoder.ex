defmodule Scholar.Preprocessing.OneHotEncoder do
  @moduledoc """
  Implements encoder that converts integer value (substitute of categorical data in tensors) into 0-1 vector.
  The index of 1 in the vector is aranged in sorted manner. This means that for x < y => one_index(x) < one_index(y).

  Currently the module supports only 1D tensors.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:encoder, :one_hot]}
  defstruct [:encoder, :one_hot]

  encode_schema = [
    num_classes: [
      required: true,
      type: :pos_integer,
      doc: """
      Number of classes to be encoded.
      """
    ]
  ]

  @encode_schema NimbleOptions.new!(encode_schema)

  @doc """
  Creates mapping from values into one-hot vectors.

  ## Options

  #{NimbleOptions.docs(@encode_schema)}

  ## Examples

      iex> t = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> Scholar.Preprocessing.OneHotEncoder.fit(t, num_classes: 4)
      %Scholar.Preprocessing.OneHotEncoder{
        encoder: %Scholar.Preprocessing.OrdinalEncoder{
          encoding_tensor: Nx.tensor(
            [
              [0, 2],
              [1, 3],
              [2, 4],
              [3, 56]
            ]
          )
        },
        one_hot: Nx.tensor(
          [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
          ], type: :u8
        )
      }
  """
  deftransform fit(tensor, opts \\ []) do
    fit_n(tensor, NimbleOptions.validate!(opts, @encode_schema))
  end

  defnp fit_n(tensor, opts) do
    encoder = Scholar.Preprocessing.OrdinalEncoder.fit(tensor, opts)
    one_hot = Nx.iota({opts[:num_classes]}) == Nx.iota({opts[:num_classes], 1})
    %__MODULE__{encoder: encoder, one_hot: one_hot}
  end

  @doc """
  Encode labels as a one-hot numeric tensor. All values provided to `transform/2` must be seen
  in `fit/2` function, otherwise an error occurs.

  ## Examples

      iex> t = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> enoder = Scholar.Preprocessing.OneHotEncoder.fit(t, num_classes: 4)
      iex> Scholar.Preprocessing.OneHotEncoder.transform(enoder, t)
      #Nx.Tensor<
        u8[7][4]
        [
          [0, 1, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
          [1, 0, 0, 0],
          [0, 0, 1, 0],
          [1, 0, 0, 0]
        ]
      >

      iex> t = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> enoder = Scholar.Preprocessing.OneHotEncoder.fit(t, num_classes: 4)
      iex> new_tensor = Nx.tensor([2, 3, 4, 3, 4, 56, 2])
      iex> Scholar.Preprocessing.OneHotEncoder.transform(enoder, new_tensor)
      #Nx.Tensor<
        u8[7][4]
        [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
          [1, 0, 0, 0]
        ]
      >
  """
  defn transform(%__MODULE__{encoder: encoder, one_hot: one_hot}, tensor) do
    decoded = Scholar.Preprocessing.OrdinalEncoder.transform(encoder, tensor)
    Nx.take(one_hot, decoded)
  end

  @doc """
  Apply encoding on the provided tensor directly. It's equivalent to `fit/2` and then `transform/2` on the same data.

  ## Examples

      iex> t = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> Scholar.Preprocessing.OneHotEncoder.fit_transform(t, num_classes: 4)
      #Nx.Tensor<
        u8[7][4]
        [
          [0, 1, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
          [1, 0, 0, 0],
          [0, 0, 1, 0],
          [1, 0, 0, 0]
        ]
      >
  """
  defn fit_transform(tensor, opts \\ []) do
    tensor
    |> fit(opts)
    |> transform(tensor)
  end
end
