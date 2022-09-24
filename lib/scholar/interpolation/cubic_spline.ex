defmodule Scholar.Interpolation.CubicSpline do
  @moduledoc """
  Cubic spline interpolation
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :x]}
  defstruct [:coefficients, :x]

  opts = [
    extrapolate: [
      required: false,
      default: true,
      type: :boolean,
      doc: "if false, out-of-bounds x return NaN."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a cubic spline interpolation of the given `(x, y)` points
  """
  defn train(x, y) do
    # https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation
    # Reference implementation in Scipy

    x = Nx.as_type(x, :f64)
    y = Nx.as_type(y, :f64)

    dx = x[1..-1//1] - x[0..-2//1]
    n = Nx.size(x)

    slope = (y[1..-1//1] - y[0..-2//1]) / dx

    s =
      if n == 3 do
        a =
          Nx.concatenate(
            [
              Nx.tensor([[1, 1, 0]]),
              Nx.stack([dx[1], 2 * (dx[0] + dx[1]), dx[0]]) |> Nx.new_axis(0),
              Nx.tensor([[0, 1, 1]])
            ],
            axis: 0
          )

        b = Nx.stack([2 * slope[0], 3 * (dx[0] * slope[1] + dx[1] * slope[0]), 2 * slope[1]])

        Nx.LinAlg.solve(a, b)
      else
        # n > 3 and bc = not_a_knot

        d_0 = Nx.new_axis(x[2] - x[0], 0)
        d_n = Nx.new_axis(x[-1] - x[-3], 0)

        up_diag =
          Nx.concatenate([
            d_0,
            dx[0..-2//1]
          ])

        low_diag =
          Nx.concatenate([
            dx[1..-1//1],
            d_n
          ])

        main_diag =
          Nx.concatenate([
            Nx.new_axis(dx[1], 0),
            2 * (dx[0..-2//1] + dx[1..-1//1]),
            Nx.new_axis(dx[-2], 0)
          ])

        a =
          Nx.broadcast(0, {n, n})
          |> Nx.put_diagonal(up_diag, offset: 1)
          |> Nx.put_diagonal(main_diag, offset: 0)
          |> Nx.put_diagonal(low_diag, offset: -1)

        bc_0 = ((dx[0] + 2 * d_0) * dx[1] * slope[0] + dx[0] ** 2 * slope[1]) / d_0

        bc_n = (dx[-1] ** 2 * slope[-2] + (2 * d_n + dx[-1]) * dx[-2] * slope[-1]) / d_n

        b =
          Nx.concatenate([
            Nx.new_axis(bc_0, 0),
            3 * (dx[1..-1//1] * slope[0..-2//1] + dx[0..-2//1] * slope[1..-1//1]),
            Nx.new_axis(bc_n, 0)
          ])

        Nx.LinAlg.solve(a, b)
      end

    slope = (y[1..-1//1] - y[0..-2//1]) / dx
    t = (s[0..-2//1] + s[1..-1//1] - 2 * slope) / dx

    c_3 = t / dx
    c_2 = (slope - s[0..-2//1]) / dx - t
    c_1 = s[0..-2//1]
    c_0 = y[0..-2//1]

    c =
      Nx.concatenate(
        [Nx.new_axis(c_3, 1), Nx.new_axis(c_2, 1), Nx.new_axis(c_1, 1), Nx.new_axis(c_0, 1)],
        axis: 1
      )

    %__MODULE__{coefficients: c, x: x}
  end

  @doc """
  Returns the value fit by `train/2` corresponding to the `target_x` input

  ### Options

  #{NimbleOptions.docs(@opts_schema)}
  """
  deftransform predict(%__MODULE__{} = model, target_x, opts \\ []) do
    predict_n(model, target_x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp predict_n(%__MODULE__{x: x, coefficients: coefficients}, target_x, opts \\ []) do
    nan_selector =
      Nx.logical_and(opts[:extrapolate] == false, target_x < x[0] or target_x > x[-1])

    nan = Nx.tensor(:nan, type: :f64)
    target_x = Nx.as_type(target_x, :f64)

    idx_selector = Nx.new_axis(target_x, 1) > Nx.new_axis(x, 0)

    idx_poly =
      idx_selector
      |> Nx.argmax(axis: 1, tie_break: :high)
      |> Nx.min(Nx.size(x) - 2)

    # deal with the case where no valid index is found
    # means that we're in the first interval
    # _poly suffix because we're selecting a specific polynomial
    # for each target_x value
    idx_poly =
      Nx.all(idx_selector == 0, axes: [1])
      |> Nx.select(0, idx_poly)

    coef_poly = Nx.take(coefficients, idx_poly)

    # each polynomial is calculated as if the origin was moved to the
    # x value that represents the start of the interval
    x_poly = target_x - Nx.take(x, idx_poly)

    result =
      x_poly
      |> Nx.new_axis(1)
      |> Nx.power(Nx.tensor([3, 2, 1, 0]))
      |> Nx.dot([1], [0], coef_poly, [1], [0])

    if opts[:extrapolate] do
      result
    else
      Nx.select(nan_selector, nan, result)
    end
  end
end
