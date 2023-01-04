defmodule Scholar.Interpolation.Linear do
  @moduledoc ~S"""
  Linear interpolation.

  This kind of interpolation is calculated by fitting polynomials
  of the first degree between each pair of given points.

  This means that for points $(x_0, y_0), (x_1, y_1)$, the
  predictive polynomial will be given by:
  $$
  \begin{cases}
  y = ax + b \newline
  a = \dfrac{y_1 - y_0}{x_1 - x_0} \newline
  b = y_1 - ax_1 = y_0 - ax_0
  \end{cases}
  $$
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :x]}
  defstruct [:coefficients, :x]

  @doc """
  Fits a linear interpolation of the given `(x, y)` points

  Inputs are expected to be rank-1 tensors with the same shape
  and at least 2 entries.

  ## Examples

      iex> x = Nx.iota({3})
      iex> y = Nx.tensor([2.0, 0.0, 1.0])
      iex> Scholar.Interpolation.Linear.fit(x, y)
      %Scholar.Interpolation.Linear{
        coefficients: #Nx.Tensor<
          f32[2][2]
          [
            [-2.0, 2.0],
            [1.0, -1.0]
          ]
        >,
        x: #Nx.Tensor<
          s64[2]
          [0, 1]
        >
      }
  """
  defn fit(x, y) do
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
  Returns the value fit by `train/2` corresponding to the `target_x` input.

  ## Examples

      iex> x = Nx.iota({3})
      iex> y = Nx.tensor([2.0, 0.0, 1.0])
      iex> model = Scholar.Interpolation.Linear.fit(x, y)
      iex> Scholar.Interpolation.Linear.predict(model, Nx.tensor([[1.0, 4.0], [3.0, 7.0]]))
      #Nx.Tensor<
        f32[2][2]
        [
          [0.0, 3.0],
          [2.0, 6.0]
        ]
      >
  """
  defn predict(%__MODULE__{x: x, coefficients: coefficients} = _model, target_x) do
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
