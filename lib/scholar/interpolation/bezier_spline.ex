defmodule Scholar.Interpolation.BezierSpline do
  @moduledoc """
  Cubic Bezier Spline interpolation.

  This kind of interpolation is calculated by fitting a set of
  continuous cubic polynomials of which the first and second
  derivatives are also continuous (which makes them curves of class $C^2$).
  In contrast to the `Scholar.Interpolation.CubicSpline` algorithm,
  the Bezier curves aren't necessarily of class $C^2$, but this interpolation
  forces this so it yields a smooth function as the result.

  The interpolated curve also has local control,
  meaning that a point that would be problematic for other
  interpolations, such as the `Scholar.Interpolation.CubicSpline`
  or `Scholar.Interpolation.Linear` algorithms, will only affect
  the segments right next to it, instead of affecting the curve as a whole.

  Computing Bezier curve is $O(N^2)$ where $N$ is the number of points.

  Reference:

    * [1] - [Bezier theory](https://en.wikipedia.org/wiki/B%C3%A9zier_curve)
    * [2] - [Spline equation derivation](https://www.particleincell.com/2012/bezier-splines/)
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :k]}
  defstruct [:coefficients, :k]

  @doc """
  Fits a cubic Bezier spline interpolation of the given `(x, y)` points.

  Inputs are expected to be rank-1 tensors with the same shape
  and at least 4 entries.

  ## Examples

      iex> x = Nx.iota({4})
      iex> y = Nx.tensor([2.0, 0.0, 1.0, 0.5])
      iex> Scholar.Interpolation.BezierSpline.fit(x, y)
      %Scholar.Interpolation.BezierSpline{
        coefficients: Nx.tensor(
          [
            [
              [0.0, 2.0],
              [0.3333335816860199, 1.033333420753479],
              [0.6666669845581055, 0.06666680425405502],
              [1.0, 0.0]
            ],
            [
              [1.0, 0.0],
              [1.3333330154418945, -0.06666680425405502],
              [1.6666665077209473, 0.7666666507720947],
              [2.0, 1.0]
            ],
            [
              [2.0, 1.0],
              [2.3333334922790527, 1.2333333492279053],
              [2.6666667461395264, 0.8666666746139526],
              [3.0, 0.5]
            ]
          ]
        ),
        k: Nx.tensor(
          [
            [0.0, 2.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.5]
          ]
        )
      }
  """
  defn fit(x, y) do
    n =
      case Nx.shape(x) do
        {x_len} when x_len > 2 ->
          x_len - 1

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

    d_low = Nx.concatenate([Nx.broadcast(1, {n - 2}), Nx.tensor([2])])
    d_main = Nx.concatenate([Nx.tensor([2]), Nx.broadcast(4, {n - 2}), Nx.tensor([7])])
    d_up = Nx.broadcast(1, {n - 1})

    a =
      Nx.broadcast(0, {n, n})
      |> Nx.put_diagonal(d_low, offset: -1)
      |> Nx.put_diagonal(d_main)
      |> Nx.put_diagonal(d_up, offset: 1)

    k = Nx.stack([x, y], axis: 1)

    b = 2 * (2 * k[0..(n - 1)] + k[1..n])

    b_comp = Nx.tensor([[0, 0], [0, 1], [0, 0], [0, 1]])
    b_comp = Nx.indexed_put(b_comp, Nx.tensor([[2, 0], [3, 0]]), Nx.broadcast(n - 1, {2}))

    b =
      Nx.indexed_put(
        b,
        b_comp,
        Nx.concatenate([k[0] + 2 * k[1], 8 * k[n - 1] + k[n]])
      )

    p1 = Nx.LinAlg.solve(a, b)

    p2 =
      Nx.concatenate([
        2 * k[1..(n - 1)] - p1[1..(n - 1)],
        Nx.new_axis((p1[n - 1] + k[n]) / 2, 0)
      ])

    coefficients =
      Nx.stack(
        [
          k[0..-2//1],
          p1,
          p2,
          k[1..-1//1]
        ],
        axis: 1
      )

    %__MODULE__{coefficients: coefficients, k: k}
  end

  predict_opts = [
    max_iter: [
      required: false,
      default: 15,
      type: :pos_integer,
      doc:
        "determines the maximum iterations for converging the t parameter for each spline polynomial"
    ],
    eps: [
      required: false,
      default: 1.0e-6,
      type: :float,
      doc:
        "determines the tolerance to be used for converging the t parameter for each spline polynomial"
    ]
  ]

  @predict_opts_schema NimbleOptions.new!(predict_opts)
  @doc """
  Returns the value fit by `fit/2` corresponding to the `target_x` input.

  ### Options

  #{NimbleOptions.docs(@predict_opts_schema)}

  ## Examples

      iex> x = Nx.iota({4})
      iex> y = Nx.tensor([2.0, 0.0, 1.0, 0.5])
      iex> model = Scholar.Interpolation.BezierSpline.fit(x, y)
      iex> Scholar.Interpolation.BezierSpline.predict(model, Nx.tensor([3.0, 4.0, 2.0, 7.0]))
      Nx.tensor(
        [0.5000335574150085, -4.2724612285383046e-5, 0.9999786615371704, 34.5]
      )
  """
  deftransform predict(%__MODULE__{} = model, target_x, opts \\ []) do
    predict_n(model, target_x, NimbleOptions.validate!(opts, @predict_opts_schema))
  end

  defnp predict_n(%__MODULE__{coefficients: coefficients, k: k}, target_x, opts) do
    input_shape = Nx.shape(target_x)
    x_poly = Nx.flatten(target_x)

    x = k[[.., 0]]
    idx_selector = Nx.new_axis(x_poly, 1) > Nx.new_axis(x, 0)

    idx_poly =
      idx_selector
      |> Nx.argmax(axis: 1, tie_break: :high)
      |> Nx.min(Nx.size(x) - 2)

    # deal with the case where no valid index is found
    # means that we're in the first interval
    # _poly suffix because we're selecting a specific polynomial
    # for each x_poly value
    idx_poly =
      Nx.all(idx_selector == 0, axes: [1])
      |> Nx.select(0, idx_poly)

    coef_poly = Nx.take(coefficients, idx_poly)

    x_curr = Nx.take(x, idx_poly)
    x_next = Nx.take(x, idx_poly + 1)

    t = t_from_x(x_poly, x_curr, x_next, coef_poly, opts)

    # polynomial_at_t returns pairs of [x, y] points.
    # we slice to get only the y values
    result = polynomial_at_t(t, coef_poly)[[.., 1]]
    Nx.reshape(result, input_shape)
  end

  defnp t_from_x(x_poly, x_curr, x_next, coef_poly, opts) do
    # for each polynomial, we need to transform x_poly into t
    # the mapping from x to t isn't necessarily linear.
    # so we first guess the initial t as the linear counterpart
    # and then we perform some iterations of gradient descent
    # followed by some iterations of binary search for the t
    # entries that haven't converged yet.
    t = (x_poly - x_curr) / (x_next - x_curr)
    t_min = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(t)), t)
    t_max = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(t)), t)

    eps = opts[:eps]
    max_iter = opts[:max_iter]

    {{t_min, t_max, t, value}, _} =
      while {{t_min, t_max, t, _value = x_poly},
             {update_mask = Nx.broadcast(1, t), x_poly, eps, coef_poly, i = 0}},
            i < max_iter and Nx.any(update_mask) do
        # if update_mask[i] = 1, we update the entry, otherwise, we keep it
        value = polynomial_at_t(t, coef_poly)[[.., 0]]

        derivative = (polynomial_at_t(t + eps, coef_poly)[[.., 0]] - value) / eps

        convergence_selector = Nx.abs(value - x_poly) < eps or Nx.abs(derivative) < eps

        update_mask = Nx.select(convergence_selector, 0, update_mask)

        t_min = Nx.select(value < x_poly, t, t_min)
        t_max = Nx.select(value < x_poly, t_max, t)

        t =
          Nx.select(
            update_mask,
            t - (value - x_poly) / Nx.select(Nx.abs(derivative) < eps, 1, derivative),
            t
          )

        {{t_min, t_max, t, value}, {update_mask, x_poly, eps, coef_poly, i + 1}}
      end

    {t, _} =
      while {t,
             {t_min, t_max, value, x_poly, coef_poly, eps, update_mask = Nx.broadcast(1, t),
              i = 0}},
            i < max_iter and Nx.any(update_mask) do
        upd_selector = value < x_poly and update_mask
        t_min = Nx.select(upd_selector, t, t_min)
        t_max = Nx.select(upd_selector, t_max, t)
        t = Nx.select(upd_selector, (t + t_max) / 2, (t + t_min) / 2)
        value = polynomial_at_t(t, coef_poly)[[.., 0]]

        update_mask = Nx.select(Nx.abs(value - x_poly) > eps, update_mask, 0)
        {t, {t_min, t_max, value, x_poly, coef_poly, eps, update_mask, i + 1}}
      end

    t
  end

  defnp polynomial_at_t(t, coef_poly) do
    # t is a {n}-shaped tensor (n can be 1, though)
    t_poly = Nx.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], axis: 1)
    Nx.dot(t_poly, [1], [0], coef_poly, [1], [0])
  end
end
