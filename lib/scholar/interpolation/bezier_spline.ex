defmodule Scholar.Interpolation.BezierSpline do
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :k]}
  defstruct [:coefficients, :k]

  defn fit(x, y) do
    {x_len} = Nx.shape(x)
    n = x_len - 1
    d_low = Nx.concatenate([Nx.broadcast(1, {n - 2}), Nx.tensor([2])])
    d_main = Nx.concatenate([Nx.tensor([2]), Nx.broadcast(4, {n - 2}), Nx.tensor([7])])
    d_up = Nx.broadcast(1, {n - 1})

    a =
      Nx.broadcast(0, {n, n})
      |> Nx.put_diagonal(d_low, offset: -1)
      |> Nx.put_diagonal(d_main)
      |> Nx.put_diagonal(d_up, offset: 1)

    k = Nx.stack([x, y], axis: 1)

    # b =
    #   Nx.concatenate([
    #     Nx.new_axis(k[0] + 2 * k[1], 0),
    #     2 * k[1..(n - 2)] + 4 * k[2..(n - 2)],
    #     Nx.new_axis(8 * k[n - 1] + k[n], 0)
    #   ])

    b = 2 * (2 * k[0..(n - 1)] + k[1..n])

    b =
      Nx.indexed_put(
        b,
        Nx.tensor([[0, 0], [0, 1], [n - 1, 0], [n - 1, 1]]),
        Nx.concatenate([k[0] + 2 * k[1], 8 * k[n - 1] + k[n]])
      )

    #  P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    # P[0] = points[0] + 2 * points[1]
    # P[n - 1] = 8 * points[n - 1] + points[n]

    p1 =
      Nx.LinAlg.solve(print_value(a, label: "a"), print_value(b, label: "b"))
      |> print_value(label: "p1")

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

  defn predict(%__MODULE__{coefficients: coefficients, k: k} = model, target_x) do
    input_shape = Nx.shape(target_x)
    x_poly = Nx.flatten(target_x)

    x = k[[0..-1//1, 0]]
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

    # for each polynomial, we need to transform x_poly into t
    # through t := (x_poly - x_i) / (x_i+1 - x_i), so t is defined
    # in the interval [0, 1], because each polynomial is defined in
    # for the parameterized t in that domain

    x_curr = Nx.take(x, idx_poly)
    x_next = Nx.take(x, idx_poly + 1)

    t = t_from_x(model, x_poly, x_curr, x_next, coef_poly)

    # https://www.particleincell.com/2012/bezier-splines/

    result = point_from_t(t, coef_poly)[[0..-1//1, 1]]
    Nx.reshape(result, input_shape)
  end

  defnp t_from_x(model, x_poly, x_curr, x_next, coef_poly) do
    t = (x_poly - x_curr) / (x_next - x_curr)
    t_min = Nx.broadcast(0.0, t)
    t_max = Nx.broadcast(1.0, t)

    eps = 1.0e-6

    {_upd_mask, t_min, t_max, t, value, _x_poly, _eps, _coef_poly, _i} =
      while {update_mask = Nx.broadcast(1, t), t_min, t_max, t, value = x_poly, x_poly, eps,
             coef_poly, i = 0},
            i < 8 do
        # if update_mask[i] = 1, we update the entry, otherwise, we keep it
        value = point_from_t(t, coef_poly)[[0..-1//1, 0]]

        derivative = (point_from_t(t + eps, coef_poly)[[0..-1//1, 0]] - value) / eps

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

        {update_mask, t_min, t_max, t, value, x_poly, eps, coef_poly, i + 1}
      end

    # for (var i = 0; Math.abs(value - xVal) > epsilon && i < 8; i++) {
    #   if (value < xVal) {
    #     tMin = t;
    #     t = (t + tMax) / 2;
    #   } else {
    #     tMax = t;
    #     t = (t + tMin) / 2;
    #   }
    #   value = this.getPointX(t);
    # }

    {_t_min, _t_max, t, _value, _x_poly, _coef_poly, _eps, _update_mask, _i} =
      while {t_min, t_max, t, value, x_poly, coef_poly, eps, update_mask = Nx.broadcast(1, t),
             i = 0},
            i < 8 do
        upd_selector = value < x_poly and update_mask
        t_min = Nx.select(upd_selector, t, t_min)
        t_max = Nx.select(upd_selector, t_max, t)
        t = Nx.select(upd_selector, (t + t_max) / 2, (t + t_min) / 2)
        value = point_from_t(t, coef_poly)[[0..-1//1, 0]]

        update_mask = Nx.select(Nx.abs(value - x_poly) > eps, update_mask, 0)

        {t_min, t_max, t, value, x_poly, coef_poly, eps, update_mask, i + 1}
      end

    t
  end

  defn point_from_t(t, coef_poly) do
    t_poly = Nx.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], axis: 1)
    Nx.dot(t_poly, [1], [0], coef_poly, [1], [0])
  end

  # returns the t parameters and the corresponding coefficient index
  defn as_poly_params(%__MODULE__{k: k}, target_x) do
    input_shape = Nx.shape(target_x)
    x_poly = Nx.flatten(target_x)

    x = k[[0..-1//1, 0]]
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

    # for each polynomial, we need to transform x_poly into t
    # through t := (x_poly - x_i) / (x_i+1 - x_i), so t is defined
    # in the interval [0, 1], because each polynomial is defined in
    # for the parameterized t in that domain

    x_curr = Nx.take(x, idx_poly)
    x_next = Nx.take(x, idx_poly + 1)
    t = (x_poly - x_curr) / (x_next - x_curr)

    # https://www.particleincell.com/2012/bezier-splines/
    {Nx.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], axis: 1)
     |> Nx.reshape(to_out_shape(input_shape)), idx_poly}
  end

  deftransformp(to_out_shape(input_shape), do: Tuple.append(input_shape, 4))
end
