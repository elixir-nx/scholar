defmodule Scholar.Shared do
  @moduledoc false

  # Collection of private helper functions and
  # macros for enforcing shape/type constraints,
  # and doing shape calculations.

  import Nx.Defn
  require Nx

  @doc """
  Asserts `left` has same shape as `right`.
  """
  deftransform assert_same_shape!(left, right) do
    left_shape = Nx.shape(left)
    right_shape = Nx.shape(right)

    unless left_shape == right_shape do
      raise ArgumentError,
            "expected tensor to have shape #{inspect(left_shape)}, got tensor with shape #{inspect(right_shape)}"
    end
  end

  @doc """
  Asserts `tensor` has rank `target_rank`.
  """
  deftransform assert_rank!(tensor, target_rank) do
    rank = Nx.rank(tensor)

    unless rank == target_rank do
      raise ArgumentError,
            "expected tensor to have rank #{target_rank}, got tensor with rank #{rank}"
    end
  end

  @doc """
  Returns the floating type of `tensor`.
  """
  deftransform to_float_type(tensor) do
    tensor |> Nx.type() |> Nx.Type.to_floating()
  end

  @doc """
  Converts `tensor` to the floating type.
  """
  defn to_float(tensor) do
    type = to_float_type(tensor)
    Nx.as_type(tensor, type)
  end

  deftransform validate_weights(weights, num_samples, opts \\ []) do
    type = opts[:type]

    cond do
      is_nil(weights) ->
        Nx.tensor(1.0, type: type)

      Nx.is_tensor(weights) and Nx.shape(weights) in [{}, {num_samples}] ->
        weights |> Nx.broadcast({num_samples}) |> Nx.as_type(type)

      is_list(weights) and length(weights) == num_samples ->
        Nx.tensor(weights, type: type)

      true ->
        raise ArgumentError,
              "invalid value for :weights option: expected list or tensor of positive numbers of size #{num_samples}, got: #{inspect(weights)}"
    end
  end

  deftransform valid_broadcast!(n_dims, shape1, shape2) do
    if tuple_size(shape1) != tuple_size(shape2) do
      raise ArgumentError,
            "expected shapes to have same rank, got #{inspect(tuple_size(shape1))} and #{inspect(tuple_size(shape2))}"
    end

    valid_broadcast(n_dims, n_dims, shape1, shape2)
  end

  deftransformp valid_broadcast(0, _n_dims, _shape1, _shape2), do: true

  deftransformp valid_broadcast(to_parse, n_dims, shape1, shape2) do
    dim1 = elem(shape1, n_dims - to_parse)
    dim2 = elem(shape2, n_dims - to_parse)

    if not (dim1 == 1 or dim2 == 1 or dim2 == dim1) do
      raise ArgumentError,
            "tensors must be broadcast compatible, got tensors with shapes #{inspect(shape1)} and #{inspect(shape2)}"
    end

    valid_broadcast(to_parse - 1, n_dims, shape1, shape2)
  end
end
