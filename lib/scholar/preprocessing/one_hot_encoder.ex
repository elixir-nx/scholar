defmodule Scholar.Preprocessing.OneHotEncoder do
  @moduledoc """
  Implements encoder that converts integer value (substitute of categorical data in tensors) into 0-1 vector.
  The index of 1 in the vector is aranged in sorted manner. This means that for x < y => one_index(x) < one_index(y).

  Currently the module supports only 1D tensors.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:ordinal_encoder]}
  defstruct [:ordinal_encoder]

  encode_schema = [
    num_categories: [
      required: true,
      type: :pos_integer,
      doc: """
      The number of categories to be encoded.
      """
    ]
  ]

  @encode_schema NimbleOptions.new!(encode_schema)

  @doc """
  Creates mapping from values into one-hot vectors.

  ## Options

  #{NimbleOptions.docs(@encode_schema)}

  ## Examples

      iex> tensor = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> Scholar.Preprocessing.OneHotEncoder.fit(tensor, num_categories: 4)
      %Scholar.Preprocessing.OneHotEncoder{
        ordinal_encoder: %Scholar.Preprocessing.OrdinalEncoder{
          categories: Nx.tensor([2, 3, 4, 56]
          )
        }
      }
  """
  deftransform fit(tensor, opts) do
    if Nx.rank(tensor) != 1 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples}, \
            got tensor with shape: #{inspect(Nx.shape(tensor))}
            """
    end

    opts = NimbleOptions.validate!(opts, @encode_schema)

    fit_n(tensor, opts)
  end

  defnp fit_n(tensor, opts) do
    ordinal_encoder = Scholar.Preprocessing.OrdinalEncoder.fit(tensor, opts)
    %__MODULE__{ordinal_encoder: ordinal_encoder}
  end

  @doc """
  Encode labels as a one-hot numeric tensor. All values provided to `transform/2` must be seen
  in `fit/2` function, otherwise an error occurs.

  ## Examples

      iex> tensor = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> encoder = Scholar.Preprocessing.OneHotEncoder.fit(tensor, num_categories: 4)
      iex> Scholar.Preprocessing.OneHotEncoder.transform(encoder, tensor)
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

      iex> tensor = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> encoder = Scholar.Preprocessing.OneHotEncoder.fit(tensor, num_categories: 4)
      iex> new_tensor = Nx.tensor([2, 3, 4, 3, 4, 56, 2])
      iex> Scholar.Preprocessing.OneHotEncoder.transform(encoder, new_tensor)
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
  defn transform(%__MODULE__{ordinal_encoder: ordinal_encoder}, tensor) do
    num_categories = Nx.size(ordinal_encoder.categories)
    num_samples = Nx.size(tensor)

    encoded =
      ordinal_encoder
      |> Scholar.Preprocessing.OrdinalEncoder.transform(tensor)
      |> Nx.new_axis(1)
      |> Nx.broadcast({num_samples, num_categories})

    encoded == Nx.iota({num_samples, num_categories}, axis: 1)
  end

  @doc """
  Appl
   encoding on the provided tensor directly. It's equivalent to `fit/2` and then `transform/2` on the same data.

  ## Examples

      iex> tensor = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> Scholar.Preprocessing.OneHotEncoder.fit_transform(tensor, num_categories: 4)
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
  deftransform fit_transform(tensor, opts) do
    if Nx.rank(tensor) != 1 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples}, \
            got tensor with shape: #{inspect(Nx.shape(tensor))}
            """
    end

    opts = NimbleOptions.validate!(opts, @encode_schema)
    fit_transform_n(tensor, opts)
  end

  defnp fit_transform_n(tensor, opts) do
    num_samples = Nx.size(tensor)
    num_categories = opts[:num_categories]

    encoded =
      tensor
      |> Scholar.Preprocessing.OrdinalEncoder.fit_transform()
      |> Nx.new_axis(1)
      |> Nx.broadcast({num_samples, num_categories})

    encoded == Nx.iota({num_samples, num_categories}, axis: 1)
  end
end
