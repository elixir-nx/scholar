defmodule Scholar.Interpolation.CubicSpline do
  @moduledoc """
  Cubic Spline interpolation.

  This kind of interpolation is calculated by fitting a set of
  continuous cubic polynomials of which the first and second
  derivatives are also continuous.

  The interpolated curve is smooth and mitigates oscillations that
  could appear if a single n-th degree polynomial were to be
  fitted over all of the points.

  Cubic spline interpolation is $O(N)$ where $N$ is the number of points.

  Reference:

    * [1] - [Cubic Spline Interpolation theory](https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation)
    * [2] - [SciPy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:coefficients, :x]}
  defstruct [:coefficients, :x]

  fit_opts = [
    boundary_condition: [
      required: false,
      default: :not_a_knot,
      type: :atom,
      doc: "one of :not_a_knot or :natural"
    ]
  ]

  @fit_opts_schema NimbleOptions.new!(fit_opts)

  @doc """
  Fits a cubic spline interpolation of the given `(x, y)` points

  Inputs are expected to be rank-1 tensors with the same shape
  and at least 3 entries.

  ### Options

  #{NimbleOptions.docs(@fit_opts_schema)}

  ## Examples

      iex> x = Nx.iota({3})
      iex> y = Nx.tensor([2.0, 0.0, 1.0])
      iex> Scholar.Interpolation.CubicSpline.fit(x, y)
      %Scholar.Interpolation.CubicSpline{
        coefficients: Nx.tensor(
          [
            [0.0, 1.5, -3.5, 2.0],
            [0.0, 1.5, -0.5, 0.0]
          ]
        ),
        x: Nx.tensor(
          [0, 1, 2]
        )
      }
  """
  deftransform fit(x, y, opts \\ []) do
    fit_n(x, y, NimbleOptions.validate!(opts, @fit_opts_schema))
  end

  defnp fit_n(x, y, opts) do
    # https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation
    # Reference implementation in Scipy

    n =
      case Nx.shape(x) do
        {n} when n > 2 ->
          n

        shape ->
          raise ArgumentError,
                "expected x to be a tensor with shape {n}, where n > 2, got: #{inspect(shape)}"
      end

    case {Nx.shape(y), Nx.shape(x)} do
      {shape, shape} ->
        :ok

      {y_shape, x_shape} ->
        raise ArgumentError,
              "expected y to have shape #{inspect(x_shape)}, got: #{inspect(y_shape)}"
    end

    dx = Nx.diff(x)

    sort_idx = Nx.argsort(x)
    x = Nx.take(x, sort_idx)
    y = Nx.take(y, sort_idx)

    dy = Nx.diff(y)

    slope = dy / dx

    s =
      case {n, opts[:boundary_condition]} do
        {3, :not_a_knot} ->
          a =
            Nx.stack([
              Nx.tensor([1, 1, 0]),
              Nx.stack([dx[1], 2 * (dx[0] + dx[1]), dx[0]]),
              Nx.tensor([0, 1, 1])
            ])

          b = Nx.stack([2 * slope[0], 3 * (dx[0] * slope[1] + dx[1] * slope[0]), 2 * slope[1]])

          tridiagonal_solve(a, b)

        {_, :not_a_knot} ->
          up_diag =
            Nx.concatenate([
              Nx.new_axis(x[2] - x[0], 0),
              dx[0..-2//1]
            ])

          low_diag =
            Nx.concatenate([
              dx[1..-1//1],
              Nx.new_axis(x[-1] - x[-3], 0)
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

          d = x[2] - x[0]
          b_0 = ((dx[0] + 2 * d) * dx[1] * slope[0] + dx[0] ** 2 * slope[1]) / d

          d = x[-1] - x[-3]
          b_n = (dx[-1] ** 2 * slope[-2] + (2 * d + dx[-1]) * dx[-2] * slope[-1]) / d

          b =
            Nx.concatenate([
              Nx.new_axis(b_0, 0),
              3 * (dx[1..-1//1] * slope[0..-2//1] + dx[0..-2//1] * slope[1..-1//1]),
              Nx.new_axis(b_n, 0)
            ])

          tridiagonal_solve(a, b)

        _ ->
          up_diag =
            Nx.concatenate([
              Nx.new_axis(dx[0], 0),
              dx[0..-2//1]
            ])

          low_diag =
            Nx.concatenate([
              dx[1..-1//1],
              Nx.new_axis(dx[-1], 0)
            ])

          main_diag =
            Nx.concatenate([
              Nx.new_axis(2 * dx[0], 0),
              2 * (dx[0..-2//1] + dx[1..-1//1]),
              Nx.new_axis(2 * dx[-1], 0)
            ])

          a =
            Nx.broadcast(0, {n, n})
            |> Nx.put_diagonal(up_diag, offset: 1)
            |> Nx.put_diagonal(main_diag, offset: 0)
            |> Nx.put_diagonal(low_diag, offset: -1)

          b_0 = 3 * (y[1] - y[0])

          b_n = 3 * (y[-1] - y[-2])

          b =
            Nx.concatenate([
              Nx.new_axis(b_0, 0),
              3 * (dx[1..-1//1] * slope[0..-2//1] + dx[0..-2//1] * slope[1..-1//1]),
              Nx.new_axis(b_n, 0)
            ])

          tridiagonal_solve(a, b)
      end

    t = (s[0..-2//1] + s[1..-1//1] - 2 * slope) / dx

    c_3 = t / dx
    c_2 = (slope - s[0..-2//1]) / dx - t
    c_1 = s[0..-2//1]
    c_0 = y[0..-2//1]

    c = Nx.stack([c_3, c_2, c_1, c_0], axis: 1)

    %__MODULE__{coefficients: c, x: x}
  end

  predict_opts = [
    extrapolate: [
      required: false,
      default: true,
      type: :boolean,
      doc: "if false, out-of-bounds x return NaN."
    ]
  ]

  @predict_opts_schema NimbleOptions.new!(predict_opts)

  @doc """
  Returns the value fit by `fit/2` corresponding to the `target_x` input

  ### Options

  #{NimbleOptions.docs(@predict_opts_schema)}

  ## Examples

      iex> x = Nx.iota({3})
      iex> y = Nx.tensor([2.0, 0.0, 1.0])
      iex> model = Scholar.Interpolation.CubicSpline.fit(x, y)
      iex> Scholar.Interpolation.CubicSpline.predict(model, Nx.tensor([[1.0, 4.0], [3.0, 7.0]]))
      Nx.tensor(
        [
          [0.0, 12.0],
          [5.0, 51.0]
        ]
      )
  """
  deftransform predict(%__MODULE__{} = model, target_x, opts \\ []) do
    predict_n(model, target_x, NimbleOptions.validate!(opts, @predict_opts_schema))
  end

  defnp predict_n(%__MODULE__{x: x, coefficients: coefficients}, target_x, opts) do
    original_shape = Nx.shape(target_x)

    target_x =
      case target_x do
        {} ->
          Nx.new_axis(target_x, 0)

        _ ->
          Nx.flatten(target_x)
      end

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
      |> Nx.pow(Nx.tensor([3, 2, 1, 0]))
      |> Nx.dot([1], [0], coef_poly, [1], [0])

    result =
      if opts[:extrapolate] do
        result
      else
        nan_selector = target_x < x[0] or target_x > x[-1]

        nan = Nx.tensor(:nan, type: to_float_type(target_x))
        Nx.select(nan_selector, nan, result)
      end

    Nx.reshape(result, original_shape)
  end

  defnp tridiagonal_solve(a, b) do
    n = Nx.size(b)
    w = Nx.broadcast(0, {n - 1})
    p = g = Nx.broadcast(0, {n})
    i = Nx.take_diagonal(a, offset: -1)
    j = Nx.take_diagonal(a)
    k = Nx.take_diagonal(a, offset: 1)

    w_0 = k[0] / j[0]
    g_0 = b[0] / j[0]
    w = Nx.indexed_put(w, Nx.new_axis(0, 0), w_0)
    g = Nx.indexed_put(g, Nx.new_axis(0, 0), g_0)

    {{w, g}, _} =
      while {{w, g}, {index = 1, i, j, k, b}}, index < n do
        w =
          if index < n - 1 do
            w_i = k[index] / (j[index] - i[index - 1] * w[index - 1])
            Nx.indexed_put(w, Nx.new_axis(index, 0), w_i)
          else
            w
          end

        g_i = (b[index] - i[index - 1] * g[index - 1]) / (j[index] - i[index - 1] * w[index - 1])
        g = Nx.indexed_put(g, Nx.new_axis(index, 0), g_i)

        {{w, g}, {index + 1, i, j, k, b}}
      end

    p = Nx.indexed_put(p, Nx.new_axis(n - 1, 0), g[n - 1])

    {p, _} =
      while {p, {index = n - 1, g, w}}, index > 0 do
        p_i = g[index - 1] - w[index - 1] * p[index]
        p = Nx.indexed_put(p, Nx.new_axis(index - 1, 0), p_i)

        {p, {index - 1, g, w}}
      end

    p
  end
end
