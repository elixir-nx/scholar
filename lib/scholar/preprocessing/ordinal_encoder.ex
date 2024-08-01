defmodule Scholar.Preprocessing.OrdinalEncoder do
  @moduledoc """
  Implements encoder that converts integer value (substitute of categorical data in tensors) into other integer value.
  The values assigned starts from `0` and go up to `num_categories - 1`. They are maintained in sorted manner.
  This means that for x < y => encoded_value(x) < encoded_value(y).

  Currently the module supports only 1D tensors.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:categories]}
  defstruct [:categories]

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
  Fit the ordinal encoder to provided data. The labels are assigned in a sorted manner.

  ## Options

  #{NimbleOptions.docs(@encode_schema)}

  ## Examples

      iex> tensor = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> Scholar.Preprocessing.OrdinalEncoder.fit(tensor, num_categories: 4)
      %Scholar.Preprocessing.OrdinalEncoder{
        categories: Nx.tensor([2, 3, 4, 56])
      }
  """
  deftransform fit(tensor, opts \\ []) do
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
    categories =
      if Nx.size(tensor) > 1 do
        sorted = Nx.sort(tensor)
        num_categories = opts[:num_categories]

        # A mask with a single 1 in every group of equal values
        representative_mask =
          Nx.concatenate([
            sorted[0..-2//1] != sorted[1..-1//1],
            Nx.tensor([true])
          ])

        representative_indices =
          representative_mask
          |> Nx.argsort(direction: :desc, stable: true)
          |> Nx.slice_along_axis(0, num_categories)

        Nx.take(sorted, representative_indices)
      else
        tensor
      end

    %__MODULE__{categories: categories}
  end

  @doc """
  Encodes tensor elements into integers from range 0 to `:num_categories - 1` or -1 if the value did not occur in fitting process.

  ## Examples

      iex> tensor = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> encoder = Scholar.Preprocessing.OrdinalEncoder.fit(tensor, num_categories: 4)
      iex> Scholar.Preprocessing.OrdinalEncoder.transform(encoder, tensor)
      #Nx.Tensor<
        s64[7]
        [1, 0, 2, 3, 0, 2, 0]
      >

      iex> tensor = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> encoder = Scholar.Preprocessing.OrdinalEncoder.fit(tensor, num_categories: 4)
      iex> new_tensor = Nx.tensor([2, 3, 4, 5, 4, 56, 2])
      iex> Scholar.Preprocessing.OrdinalEncoder.transform(encoder, new_tensor)
      #Nx.Tensor<
        s64[7]
        [0, 1, 2, -1, 2, 3, 0]
      >
  """
  defn transform(%__MODULE__{categories: categories}, tensor) do
    if Nx.rank(tensor) != 1 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples}, \
            got tensor with shape: #{inspect(Nx.shape(tensor))}
            """
    end

    num_categories = Nx.size(categories)
    size = Nx.size(tensor)
    input_vectorized_axes = tensor.vectorized_axes

    tensor =
      Nx.revectorize(tensor, [x: size], target_shape: {:auto})

    left = 0
    right = num_categories - 1
    label = -1

    [left, right, label, tensor] =
      Nx.broadcast_vectors([
        left,
        right,
        label,
        tensor
      ])

    {label, _} =
      while {label, {left, right, tensor, categories}}, left <= right do
        curr = Nx.quotient(left + right, 2)

        cond do
          tensor[0] > categories[curr] ->
            {label, {curr + 1, right, tensor, categories}}

          tensor[0] < categories[curr] ->
            {label, {left, curr - 1, tensor, categories}}

          true ->
            {curr, {1, 0, tensor, categories}}
        end
      end

    Nx.revectorize(label, input_vectorized_axes, target_shape: {size})
  end

  @doc """
  Decodes tensor elements into original categories seen during fitting.

  ## Examples

    iex> tensor = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
    iex> encoder = Scholar.Preprocessing.OridinalEncoder.fit(tensor, num_categories: 4)
    iex> encoded = Nx.tensor([1, 0, 2, 3, 1, 0, 2])
    iex> Scholar.Preprocessing.OridinalEncoder.inverse_transform(encoder, encoded)
    Nx.tensor([3, 2, 4, 56, 3, 2, 4])
  """
  deftransform inverse_transform(%__MODULE__{categories: categories}, encoded_tensor) do
    Nx.take(categories, encoded_tensor)
  end

  @doc """
  Apply encoding on the provided tensor directly. It's equivalent to `fit/2` and then `transform/2` on the same data.

  ## Examples

      iex> tensor = Nx.tensor([3, 2, 4, 56, 2, 4, 2])
      iex> Scholar.Preprocessing.OridinalEncoder.fit_transform(tensor)
      #Nx.Tensor<
        u64[7]
        [1, 0, 2, 3, 0, 2, 0]
      >
  """
  deftransform fit_transform(tensor) do
    if Nx.rank(tensor) != 1 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples}, \
            got tensor with shape: #{inspect(Nx.shape(tensor))}
            """
    end

    fit_transform_n(tensor)
  end

  defnp fit_transform_n(tensor) do
    size = Nx.size(tensor)
    indices = Nx.argsort(tensor, type: :u64)
    sorted = Nx.take(tensor, indices)

    change_indices =
      Nx.concatenate([
        Nx.tensor([true]),
        sorted[0..(size - 2)] != sorted[1..(size - 1)]
      ])

    ordinal_values =
      change_indices
      |> Nx.as_type(:u64)
      |> Nx.cumulative_sum()
      |> Nx.subtract(1)

    inverse =
      Nx.indexed_put(
        Nx.broadcast(Nx.u64(0), {size}),
        Nx.new_axis(indices, 1),
        Nx.iota({size}, type: :u64)
      )

    Nx.take(ordinal_values, inverse)
  end
end
