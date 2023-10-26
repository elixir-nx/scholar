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

  Linear interpolation has $O(N)$ time and space complexity where $N$ is the number of points.
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:coefficients, :x]}
  defstruct [:coefficients, :x]

  @type t :: %Scholar.Interpolation.Linear{}

  opts_schema = [
    left: [
      type: {:or, [:float, :integer]},
      doc:
        "Value to return for values in `target_x` smaller that the smallest value in training set"
    ],
    right: [
      type: {:or, [:float, :integer]},
      doc:
        "Value to return for values in `target_x` greater that the greatest value in training set"
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)
  @doc """
  Fits a linear interpolation of the given `(x, y)` points

  Inputs are expected to be rank-1 tensors with the same shape
  and at least 2 entries.

  ## Examples

      iex> x = Nx.iota({3})
      iex> y = Nx.tensor([2.0, 0.0, 1.0])
      iex> Scholar.Interpolation.Linear.fit(x, y)
      %Scholar.Interpolation.Linear{
        coefficients: Nx.tensor(
          [
            [-2.0, 2.0],
            [1.0, -1.0]
          ]
        ),
        x: Nx.tensor(
          [0, 1, 2]
        )
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

    %__MODULE__{coefficients: coefficients, x: x}
  end

  @doc """
  Returns the value fit by `fit/2` corresponding to the `target_x` input.

  ## Examples

      iex> x = Nx.iota({3})
      iex> y = Nx.tensor([2.0, 0.0, 1.0])
      iex> model = Scholar.Interpolation.Linear.fit(x, y)
      iex> Scholar.Interpolation.Linear.predict(model, Nx.tensor([[1.0, 4.0], [3.0, 7.0]]))
      Nx.tensor(
        [
          [0.0, 3.0],
          [2.0, 6.0]
        ]
      )

      iex> x = Nx.iota({5})
      iex> y = Nx.tensor([2.0, 0.0, 1.0, 3.0, 4.0])
      iex> model = Scholar.Interpolation.Linear.fit(x, y)
      iex> target_x = Nx.tensor([-2, -1, 1.25, 3, 3.25, 5.0])
      iex> Scholar.Interpolation.Linear.predict(model, target_x, left: 0.0, right: 10.0)
      #Nx.Tensor<
        f32[6]
        [0.0, 0.0, 0.25, 3.0, 3.25, 10.0]
      >
  """
  deftransform predict(model, target_x, opts \\ []) do
    predict_n(model, target_x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp predict_n(%__MODULE__{x: x, coefficients: coefficients} = _model, target_x, opts) do
    shape = Nx.shape(target_x)

    target_x = Nx.flatten(target_x)

    indices = Nx.argsort(target_x)

    left_bound = x[0]
    right_bound = x[-1]

    target_x = Nx.sort(target_x)
    res = Nx.broadcast(Nx.tensor(0, type: to_float_type(target_x)), {Nx.axis_size(target_x, 0)})

    # while with smaller than left_bound
    {{res, i}, _} =
      while {{res, i = 0}, {x, coefficients, left_bound, target_x}},
            check_cond_left(target_x, i, left_bound) do
        val =
          case opts[:left] do
            nil ->
              coefficients[0][0] * Nx.take(target_x, i) + coefficients[0][1]

            _ ->
              opts[:left]
          end

        res = Nx.indexed_put(res, Nx.new_axis(i, -1), val)
        {{res, i + 1}, {x, coefficients, left_bound, target_x}}
      end

    {{res, i}, _} =
      while {{res, i}, {x, right_bound, coefficients, target_x, j = 0}},
            check_cond_right(target_x, i, right_bound) do
        {j, _} =
          while {j, {i, x, target_x}},
                j < Nx.axis_size(x, 0) and Nx.take(x, j) < Nx.take(target_x, i) do
            {j + 1, {i, x, target_x}}
          end

        res =
          Nx.indexed_put(
            res,
            Nx.new_axis(i, -1),
            coefficients[Nx.max(j - 1, 0)][0] * Nx.take(target_x, i) +
              coefficients[Nx.max(j - 1, 0)][1]
          )

        i = i + 1

        {{res, i}, {x, right_bound, coefficients, target_x, j}}
      end

    {res, i}

    # while with bigger than right_bound

    {res, _} =
      while {res, {x, coefficients, target_x, i}},
            i < Nx.axis_size(target_x, 0) do
        val =
          case opts[:right] do
            nil ->
              coefficients[-1][0] * Nx.take(target_x, i) + coefficients[-1][1]

            _ ->
              opts[:right]
          end

        res = Nx.indexed_put(res, Nx.new_axis(i, -1), val)
        {res, {x, coefficients, target_x, i + 1}}
      end

    res = Nx.take(res, indices)
    Nx.reshape(res, shape)
  end

  defnp check_cond_left(target_x, i, left_bound) do
    cond do
      i >= Nx.axis_size(target_x, 0) ->
        Nx.u8(0)

      Nx.take(target_x, i) >= left_bound ->
        Nx.u8(0)

      true ->
        Nx.u8(1)
    end
  end

  defnp check_cond_right(target_x, i, right_bound) do
    cond do
      i >= Nx.axis_size(target_x, 0) ->
        Nx.u8(0)

      Nx.take(target_x, i) > right_bound ->
        Nx.u8(0)

      true ->
        Nx.u8(1)
    end
  end
end
