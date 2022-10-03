defmodule Scholar.Interpolation.Linear do
  @moduledoc """
  Linear interpolation.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :x]}
  defstruct [:coefficients, :x]

  @doc """
  Fits a linear interpolation of the given `(x, y)` points

  Inputs are expected to be rank-1 tensors with the same shape
  and at least 2 entries.
  """
  defn fit(x, y) do
    # https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation
    # Reference implementation in Scipy

    case Nx.shape(x) do
      {n} when n > 1 ->
        :ok

      shape ->
        raise ArgumentError,
              "expected x to be a tensor with shape {n}, where n > 1, got: #{inspect(shape)}"
    end

    case {Nx.shape(y), Nx.shape(x)} do
      {shape, shape} ->
        :ok

      {y_shape, x_shape} ->
        raise ArgumentError,
              "expected y to have shape #{inspect(x_shape)}, got: #{inspect(y_shape)}"
    end

    sort_idx = Nx.argsort(x)
    x = Nx.take(x, sort_idx)
    y = Nx.take(y, sort_idx)

    x0 = x[0..-2//1]
    x1 = x[1..-1//1]

    y0 = y[0..-2//1]
    y1 = y[1..-1//1]

    a = (y1 - y0) / (x1 - x0)
    b = y0 - a * x0

    coefficients = Nx.stack([a, b], axis: 1)

    %__MODULE__{coefficients: coefficients, x: x0}
  end

  @doc """
  Returns the value fit by `train/2` corresponding to the `target_x` input
  """
  defn predict(%__MODULE__{x: x, coefficients: coefficients}, target_x) do
    original_shape = Nx.shape(target_x)

    target_x = Nx.flatten(target_x)

    idx_selector = Nx.new_axis(target_x, 1) >= x

    idx_poly = Nx.argmax(idx_selector, axis: 1, tie_break: :high)

    idx_poly = Nx.select(Nx.all(idx_selector == 0, axes: [1]), 0, idx_poly)

    coef_poly = Nx.take(coefficients, idx_poly)

    x_poly = Nx.stack([target_x, Nx.broadcast(1, target_x)], axis: 1)

    result = Nx.dot(x_poly, [1], [0], coef_poly, [1], [0])

    Nx.reshape(result, original_shape)
  end
end
