defmodule Scholar.Preprocessing.OrdinalEncoder do
  @moduledoc """
  Implements encoder that converts integer value (substitute of categorical data in tensors) into other integer value.
  The values assigned starts from `0` and go up to `num_classes - 1`.They are maintained in sorted manner.
  This means that for x < y => encoded_value(x) < encoded_value(y).

  Currently the module supports only 1D tensors.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:encoding_tensor]}
  defstruct [:encoding_tensor]

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
  Fit the ordinal encoder to provided data. The labels are assigned in a sorted manner.

  ## Options

  #{NimbleOptions.docs(@encode_schema)}

  ## Examples

      iex> t = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> Scholar.Preprocessing.OrdinalEncoder.fit(t, num_classes: 4)
      %Scholar.Preprocessing.OrdinalEncoder{
        encoding_tensor: Nx.tensor(
          [
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 56]
          ]
        )
      }
  """
  deftransform fit(tensor, opts \\ []) do
    fit_n(tensor, NimbleOptions.validate!(opts, @encode_schema))
  end

  defnp fit_n(tensor, opts) do
    sorted = Nx.sort(tensor)
    num_classes = opts[:num_classes]

    # A mask with a single 1 in every group of equal values
    representative_mask =
      Nx.concatenate([
        sorted[0..-2//1] != sorted[1..-1//1],
        Nx.tensor([1])
      ])

    representative_indices =
      representative_mask
      |> Nx.argsort(direction: :desc)
      |> Nx.slice_along_axis(0, num_classes)

    representative_values = Nx.take(sorted, representative_indices) |> Nx.new_axis(-1)

    encoding_tensor =
      Nx.concatenate([Nx.iota(Nx.shape(representative_values)), representative_values], axis: 1)

    %__MODULE__{encoding_tensor: encoding_tensor}
  end

  @doc """
  Encodes a tensor's values into integers from range 0 to `:num_classes - 1` or -1 if the value did not occur in fitting process.

  ## Examples

      iex> t = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> enoder = Scholar.Preprocessing.OrdinalEncoder.fit(t, num_classes: 4)
      iex> Scholar.Preprocessing.OrdinalEncoder.transform(enoder, t)
      #Nx.Tensor<
        s64[7]
        [1, 0, 2, 3, 0, 2, 0]
      >

      iex> t = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> enoder = Scholar.Preprocessing.OrdinalEncoder.fit(t, num_classes: 4)
      iex> new_tensor = Nx.tensor([2, 3, 4, 5, 4, 56, 2])
      iex> Scholar.Preprocessing.OrdinalEncoder.transform(enoder, new_tensor)
      #Nx.Tensor<
        s64[7]
        [0, 1, 2, -1, 2, 3, 0]
      >
  """
  defn transform(%__MODULE__{encoding_tensor: encoding_tensor}, tensor) do
    tensor_size = Nx.axis_size(encoding_tensor, 0)
    decode_size = Nx.axis_size(tensor, 0)
    input_vectorized_axes = tensor.vectorized_axes

    tensor =
      Nx.revectorize(tensor, [x: decode_size], target_shape: {:auto})

    left = 0
    right = tensor_size - 1
    label = -1

    [left, right, label, tensor] =
      Nx.broadcast_vectors([
        left,
        right,
        label,
        tensor
      ])

    {label, _} =
      while {label, {left, right, tensor, encoding_tensor}}, left <= right do
        curr = Nx.quotient(left + right, 2)

        cond do
          tensor[0] > encoding_tensor[curr][1] ->
            {label, {curr + 1, right, tensor, encoding_tensor}}

          tensor[0] < encoding_tensor[curr][1] ->
            {label, {left, curr - 1, tensor, encoding_tensor}}

          true ->
            {encoding_tensor[curr][0], {1, 0, tensor, encoding_tensor}}
        end
      end

    Nx.revectorize(label, input_vectorized_axes, target_shape: {decode_size})
  end

  @doc """
  Apply encoding on the provided tensor directly. It's equivalent to `fit/2` and then `transform/2` on the same data.

  ## Examples

      iex> t = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> Scholar.Preprocessing.OridinalEncoder.fit_transform(t, num_classes: 4)
      #Nx.Tensor<
        s64[7]
        [1, 0, 2, 3, 0, 2, 0]
      >
  """
  defn fit_transform(tensor, opts \\ []) do
    tensor
    |> fit(opts)
    |> transform(tensor)
  end
end
