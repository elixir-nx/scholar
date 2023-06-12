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

      Nx.is_tensor(weights) and Nx.shape(weights) == {num_samples} ->
        Nx.as_type(weights, type)

      is_list(weights) and length(weights) == num_samples ->
        Nx.tensor(weights, type: type)

      true ->
        raise ArgumentError,
              "invalid value for :weights option: expected list or tensor of positive numbers of size #{num_samples}, got: #{inspect(weights)}"
    end
  end
end
