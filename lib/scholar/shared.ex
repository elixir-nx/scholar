defmodule Scholar.Shared do
  @moduledoc false

  # Collection of private helper functions and
  # macros for enforcing shape/type constraints,
  # and doing shape calculations.

  import Nx.Defn

  @doc """
  Asserts `lhs` has same shape as `rhs`.
  """
  deftransform assert_same_shape!(lhs, rhs) do
    lhs = Nx.shape(lhs)
    rhs = Nx.shape(rhs)

    unless lhs == rhs do
      raise ArgumentError,
            "expected input shapes to be equal," <>
              " got #{inspect(lhs)} != #{inspect(rhs)}"
    end
  end
end
