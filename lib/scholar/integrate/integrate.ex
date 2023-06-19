defmodule Scholar.Integrate do
  @moduledoc """
  Module for numerical integration.
  """

  import Nx.Defn

  @doc """
  Integrate `y` along the given axis using the chosen rule.

  ## Options

  Options are the same as for the chosen rule. See the documentation for
  specific implementations for more information.

  ## Examples

      iex> Scholar.Integrate.integrate(Nx.tensor([1, 2, 3]), :simpson, Nx.tensor([4, 5, 6]))
      #Nx.Tensor<
        f32
        4.0
      >

      iex> Scholar.Integrate.integrate(Nx.tensor([[0, 1, 2], [3, 4, 5]]), :trapezoidal, Nx.tensor([[1, 2, 3], [1, 2, 3]]))
      #Nx.Tensor<
        f32[2]
        [2.0, 8.0]
      >

      iex> Scholar.Integrate.integrate(Nx.tensor([[0, 1, 2], [3, 4, 5]]), :simpson, Nx.tensor([[1, 1, 1], [2, 2, 2]]), axis: 0)
      #Nx.Tensor<
        f32[3]
        [1.5, 2.5, 3.5]
      >
  """
  deftransform integrate(y, method, x, opts \\ []) do
    case method do
      :trapezoidal ->
        Scholar.Integrate.Trapezoidal.trapezoidal(y, x, opts)

      :simpson ->
        Scholar.Integrate.Simpson.simpson(y, x, opts)

      _ ->
        raise ArgumentError, "Unknown integration method: #{inspect(method)}"
    end
  end

  deftransform integrate_uniform(y, method, opts \\ []) do
    case method do
      :trapezoidal ->
        Scholar.Integrate.Trapezoidal.trapezoidal_uniform(y, opts)

      :simpson ->
        Scholar.Integrate.Simpson.simpson_uniform(y, opts)

      _ ->
        raise ArgumentError, "Unknown integration method: #{inspect(method)}"
    end
  end
end
