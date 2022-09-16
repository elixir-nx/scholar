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
end
