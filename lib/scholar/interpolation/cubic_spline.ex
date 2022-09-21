defmodule Scholar.Interpolation.CubicSpline do
  @moduledoc """
  Cubic spline interpolation
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :x, :y]}
  defstruct [:coefficients, :x, :y]

  opts = [
    extrapolate: [
      required: false,
      default: true,
      type: :boolean,
      doc: "if false, out-of-bounds x values raise."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)
  @doc """
  Fits a cubic spline interpolation of the given `(x, y)` points

  ## Options

  #{NimbleOptions.docs(@opts_schema)}
  """
  deftransform train(x, y, opts \\ []) do
    train_n(x, y, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp train_n(x, y, opts \\ []) do
    # https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation

    h_i = x[1..-1//1] - x[0..-2//1]
    μ_i = h_i[0..-2//1] / (h_i[0..-2//1] + h_i[1..-1//1])
    lambda_i = 1 - μ_i

    μ_i = Nx.pad(μ_i, 1, [{0, 1, 0}])
    lambda_i = Nx.pad(lambda_i, 1, [{1, 0, 0}])

    d_i = 6 * three_point_divided_difference(x, y)

    a =
      Nx.eye(Nx.size(d_i))
      |> Nx.multiply(2)
      |> Nx.put_diagonal(μ_i, offset: -1)
      |> Nx.put_diagonal(lambda_i, offset: 1)

    %__MODULE__{coefficients: Nx.LinAlg.solve(a, d_i), x: x}
  end

  defnp three_point_divided_difference(x, y) do
    x_minus = x[0..-3//1]
    x_i = x[1..-2//1]
    x_plus = x[2..-1//1]

    y_minus = y[0..-3//1]
    y_i = y[1..-2//1]
    y_plus = y[2..-1//1]

    # divided diff f[x_minus, x_i, x_plus] = (f[x_i, x_plus] - f[x_minus, x_i]) / (x_plus - x_minus)
    # divided diff f[a, b] := (f[b] - f[a] / (x_b - x_a)) = (y_b - y_a) / (x_b - x_a)
    # f[x_i, x_plus] := (y_plus - y_i) / (x_plus - x_i)
    # f[x_minus, x_i] := (y_i - y_minus) / (x_i - x_minus)

    f_plus = (y_plus - y_i) / (x_plus - x_i)
    f_minus = (y_i - y_minus) / (x_i - x_minus)

    f = (f_plus - f_minus) / (x_plus - x_minus)

    zero = Nx.tensor([0])

    Nx.concatenate([zero, f, zero])
  end

  defn predict(%__MODULE__{x: source_x, y: source_y, coefficients: coefficients}, x) do
  end
end
