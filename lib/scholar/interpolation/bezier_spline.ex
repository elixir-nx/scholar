defmodule Scholar.Interpolation.BezierSpline do
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :k]}
  defstruct [:coefficients, :k]

  defn fit(x, y) do
    {n} = Nx.shape(x)
    d_low = Nx.concatenate([Nx.broadcast(1, {n - 3}), Nx.tensor([2])])
    d_main = Nx.concatenate([Nx.tensor([2]), Nx.broadcast(4, {n - 3}), Nx.tensor([7])])
    d_up = Nx.broadcast(1, {n - 2})

    a =
      Nx.broadcast(0, {n - 1, n - 1})
      |> Nx.put_diagonal(d_low, offset: -1)
      |> Nx.put_diagonal(d_main)
      |> Nx.put_diagonal(d_up, offset: 1)

    k = Nx.stack([x, y], axis: 1)

    b =
      Nx.concatenate([
        Nx.new_axis(k[0], 0),
        4 * k[1..-3//1],
        Nx.new_axis(8 * k[-2], 0)
      ]) +
        Nx.concatenate([
          2 * k[1..-2//1],
          Nx.new_axis(k[-1], 0)
        ])

    p1 = Nx.LinAlg.solve(a, b)

    p2 =
      Nx.concatenate([
        2 * k[1..-2//1] - p1[1..-1//1],
        Nx.new_axis((p1[-1] + k[-1]) / 2, 0)
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

  defn predict(%__MODULE__{coefficients: coefficients, k: k}, target_x) do
    input_shape = Nx.shape(target_x)

    x = k[[0..-1//1, 0]]
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

    # for each polynomial, we need to transform target_x into t
    # through t := (target_x - x_i) / (x_i+1 - x_i), so t is defined
    # in the interval [0, 1]

    x_curr = Nx.take(x, idx_poly)
    x_next = Nx.take(x, idx_poly + 1)
    t = (target_x - x_curr) / (x_next - x_curr)

    # https://www.particleincell.com/2012/bezier-splines/
    t_poly = Nx.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], axis: 1)

    result = Nx.dot(t_poly, [0], [1], coef_poly, [0], [1])
    Nx.reshape(result[[0..-1//1, 1]], input_shape)
  end
end
