defmodule Scholar.Interpolation.CubicSpline do
  @moduledoc """
  Cubic spline interpolation
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :x, :y, :dx]}
  defstruct [:coefficients, :x, :y, :dx]

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
  deftransform train(x, y, _opts \\ []) do
    train_n(x, y)
    # train_n(x, y, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp train_n(x, y) do
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

    c = Nx.stack([c_3, c_2, c_1, c_0], axis: 0)

    %__MODULE__{coefficients: c, x: x, y: y, dx: dx}
  end

  defn predict(%__MODULE__{x: x, coefficients: coefficients}, target_x) do
    target_x = Nx.as_type(target_x, :f64)
    idx = Nx.sum(x < target_x)
    idx = Nx.min(idx, Nx.size(x) - 2)

    c_poly = coefficients[[0..-1//1, idx]]

    x_poly = target_x - x[idx]

    Nx.dot(x_poly ** Nx.tensor([3.0, 2, 1, 0], type: :f64), c_poly)
  end
end
