defmodule Scholar.Interpolation.MonotonicCubicSpline do
  @moduledoc """
  Monotonic cubic spline interpolation (also known as PCHIP, from
  *Piecewise Cubic Hermite Interpolating Polynomial*).

  Like `Scholar.Interpolation.CubicSpline`, this fits a piecewise cubic
  polynomial to the given points. The derivatives at each point, however,
  are chosen with the Fritsch-Carlson method so that the resulting curve is
  monotonic wherever the data is monotonic and has no overshoot around local
  extrema. In exchange, only the first derivative of the curve is guaranteed
  to be continuous (the second one is not).

  This is a good choice when the shape of the data matters more than its
  smoothness, for example to avoid interpolated values that fall outside the
  range of the surrounding samples.

  Monotonic cubic spline interpolation is $O(N)$ where $N$ is the number of points.

  References:

    * [1] - [Monotone cubic interpolation](https://en.wikipedia.org/wiki/Monotone_cubic_interpolation)
    * [2] - [Fritsch, F. N.; Carlson, R. E. (1980). "Monotone Piecewise Cubic Interpolation"](https://doi.org/10.1137/0717021)
    * [3] - [SciPy implementation (PchipInterpolator)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:coefficients, :x]}
  defstruct [:coefficients, :x]

  @doc """
  Fits a monotonic cubic spline interpolation of the given `(x, y)` points.

  Inputs are expected to be rank-1 tensors with the same shape
  and at least 2 entries.

  ## Examples

      iex> x = Nx.iota({3})
      iex> y = Nx.tensor([0.0, 1.0, 0.0])
      iex> Scholar.Interpolation.MonotonicCubicSpline.fit(x, y)
      %Scholar.Interpolation.MonotonicCubicSpline{
        coefficients: Nx.tensor(
          [
            [0.0, -1.0, 2.0, 0.0],
            [0.0, -1.0, 0.0, 1.0]
          ]
        ),
        x: Nx.tensor(
          [0, 1, 2]
        )
      }
  """
  deftransform fit(x, y) do
    fit_n(x, y)
  end

  defnp fit_n(x, y) do
    # https://en.wikipedia.org/wiki/Monotone_cubic_interpolation
    # Reference implementation in SciPy (PchipInterpolator)

    n =
      case Nx.shape(x) do
        {n} when n > 1 ->
          n

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

    dx = Nx.diff(x)
    slope = Nx.diff(y) / dx

    s = find_derivatives(dx, slope, n)

    # convert to power-basis coefficients, as in `Scholar.Interpolation.CubicSpline`
    t = (s[0..-2//1] + s[1..-1//1] - 2 * slope) / dx

    c_3 = t / dx
    c_2 = (slope - s[0..-2//1]) / dx - t
    c_1 = s[0..-2//1]
    c_0 = y[0..-2//1]

    c = Nx.stack([c_3, c_2, c_1, c_0], axis: 1)

    %__MODULE__{coefficients: c, x: x}
  end

  defnp find_derivatives(dx, slope, n) do
    case n do
      2 ->
        # a single interval is linear: both derivatives equal the secant slope
        Nx.broadcast(slope[0], {2})

      _ ->
        sign = Nx.sign(slope)
        slope_left = slope[0..-2//1]
        slope_right = slope[1..-1//1]

        # zero the derivative where the data is not locally monotonic
        # (secant slopes change sign or one is zero) to avoid overshoot
        non_monotonic? =
          sign[1..-1//1] != sign[0..-2//1] or slope_right == 0 or slope_left == 0

        dx_left = dx[0..-2//1]
        dx_right = dx[1..-1//1]

        w1 = 2 * dx_right + dx_left
        w2 = dx_right + 2 * dx_left

        # avoid dividing by a zero slope: some backends raise on `x / 0`
        # instead of returning infinity, so masking afterwards is not enough
        safe_slope_left = Nx.select(slope_left == 0, 1.0, slope_left)
        safe_slope_right = Nx.select(slope_right == 0, 1.0, slope_right)

        weighted_harmonic_mean = (w1 / safe_slope_left + w2 / safe_slope_right) / (w1 + w2)
        safe_weighted_harmonic_mean = Nx.select(non_monotonic?, 1.0, weighted_harmonic_mean)

        interior = Nx.select(non_monotonic?, 0.0, 1.0 / safe_weighted_harmonic_mean)

        first = edge_derivative(dx[0], dx[1], slope[0], slope[1])
        last = edge_derivative(dx[-1], dx[-2], slope[-1], slope[-2])

        Nx.concatenate([Nx.new_axis(first, 0), interior, Nx.new_axis(last, 0)])
    end
  end

  # one-sided three-point estimate at the boundary, capped to avoid overshoot
  defnp edge_derivative(h0, h1, m0, m1) do
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)

    opposite_sign? = Nx.sign(d) != Nx.sign(m0)
    overshoot? = Nx.sign(m0) != Nx.sign(m1) and Nx.abs(d) > 3 * Nx.abs(m0)

    d = Nx.select(opposite_sign?, 0.0, d)
    Nx.select(not opposite_sign? and overshoot?, 3 * m0, d)
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
  Returns the value fit by `fit/2` corresponding to the `target_x` input.

  ### Options

  #{NimbleOptions.docs(@predict_opts_schema)}

  ## Examples

      iex> x = Nx.iota({3})
      iex> y = Nx.tensor([0.0, 1.0, 0.0])
      iex> model = Scholar.Interpolation.MonotonicCubicSpline.fit(x, y)
      iex> Scholar.Interpolation.MonotonicCubicSpline.predict(model, Nx.tensor([0.5, 1.5]))
      Nx.tensor(
        [0.75, 0.75]
      )
  """
  deftransform predict(%__MODULE__{} = model, target_x, opts \\ []) do
    predict_n(model, target_x, NimbleOptions.validate!(opts, @predict_opts_schema))
  end

  defnp predict_n(%__MODULE__{x: x, coefficients: coefficients}, target_x, opts) do
    original_shape = Nx.shape(target_x)
    target_x = Nx.flatten(target_x)

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
end
