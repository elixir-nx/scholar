defmodule Scholar.Shared do
  @moduledoc false

  # Collection of private helper functions and
  # macros for enforcing shape/type constraints,
  # and doing shape calculations.

  import Nx.Defn

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

  deftransform check_if_positive_float(num, param_name) do
    if is_number(num) and num > 0 do
      {:ok, num}
    else
      {:error, "expected :#{param_name} to be a positive number, got: #{inspect(num)}"}
    end
  end

  deftransform check_if_non_negative_float(num, param_name) do
    if is_number(num) and num >= 0 do
      {:ok, num}
    else
      {:error, "expected :#{param_name} to be a non-negative number, got: #{inspect(num)}"}
    end
  end
end
