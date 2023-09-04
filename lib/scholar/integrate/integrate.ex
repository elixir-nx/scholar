defmodule Scholar.Integrate do
  @moduledoc """
  Module for numerical integration.
  """

  import Nx.Defn
  import Scholar.Shared

  general_trapezoidal = [
    axis: [
      type: {:custom, Scholar.Options, :axis, []},
      default: -1,
      doc: """
      Axis along which to integrate.
      """
    ],
    keep_axis: [
      type: :boolean,
      default: false,
      doc: "If set to true, the axis which is reduced is kept."
    ]
  ]

  general_simpson =
    general_trapezoidal ++
      [
        even: [
          type: {:in, [:avg, :first, :last]},
          default: :avg,
          doc: """
          If set to `:avg`, the average of the first and last interval is used in the integration.
          If set to `:first`, the first interval is used in the integration.
          If set to `:last`, the last interval is used in the integration.
          """
        ]
      ]

  uniform = [
    dx: [
      type: {:or, [:float, :integer]},
      default: 1.0,
      doc: """
      The spacing between samples.
      """
    ]
  ]

  uniform_trapezoidal_schema = general_trapezoidal ++ uniform
  uniform_simpson_schema = general_simpson ++ uniform

  @trapezoidal_schema NimbleOptions.new!(general_trapezoidal)
  @uniform_trapezoidal_schema NimbleOptions.new!(uniform_trapezoidal_schema)

  @simpson_schema NimbleOptions.new!(general_simpson)
  @uniform_simpson_schema NimbleOptions.new!(uniform_simpson_schema)

  @doc """
  Integrate `y` along the given axis using the simpson's rule.
  The integration happens in sequence along elements of `x`.

  ## Options

  #{NimbleOptions.docs(@simpson_schema)}

  ## Examples

      iex> y = Nx.tensor([1, 2, 3])
      iex> x = Nx.tensor([4, 5, 6])
      iex> Scholar.Integrate.simpson(y, x)
      #Nx.Tensor<
        f32
        4.0
      >

      iex> y = Nx.tensor([[0, 1, 2], [3, 4, 5]])
      iex> x = Nx.tensor([[1, 2, 3], [1, 2, 3]])
      iex> Scholar.Integrate.simpson(y, x)
      #Nx.Tensor<
        f32[2]
        [2.0, 8.0]
      >

      iex> y = Nx.tensor([[0, 1, 2], [3, 4, 5]])
      iex> x = Nx.tensor([[1, 1, 1], [2, 2, 2]])
      iex> Scholar.Integrate.simpson(y, x, axis: 0)
      #Nx.Tensor<
        f32[3]
        [1.5, 2.5, 3.5]
      >
  """
  deftransform simpson(y, x, opts \\ []) do
    simpson_n(y, x, NimbleOptions.validate!(opts, @simpson_schema))
  end

  defnp simpson_n(y, x, opts \\ []) do
    axis = opts[:axis]
    y_ndim = Nx.rank(y)
    axis_size = Nx.axis_size(y, axis)
    check_shape(x, y, axis)
    {avg, first, last} = parse_even(opts[:even])

    x =
      if Nx.rank(x) == 1 do
        shape = prepare_shape(y_ndim, Nx.size(x), axis)
        Nx.reshape(x, shape)
      else
        x
      end

    results = Nx.tensor(0.0, type: to_float_type(y))
    val = Nx.tensor(0.0, type: to_float_type(y))

    cond do
      rem(axis_size, 2) == 0 ->
        {results, val} =
          if avg or first do
            y_slice0 = Nx.take(y, axis_size - 1, axis: axis)
            y_slice1 = Nx.take(y, axis_size - 2, axis: axis)

            x_slice0 = Nx.take(x, axis_size - 1, axis: axis)
            x_slice1 = Nx.take(x, axis_size - 2, axis: axis)

            last_dx = x_slice0 - x_slice1

            val = val + 0.5 * last_dx * (y_slice0 + y_slice1)

            results =
              results +
                basic_simpson(y, 0, x,
                  len: if(axis_size - 3 >= 0, do: axis_size - 3, else: 3 - axis_size),
                  axis: axis,
                  keep_axis: opts[:keep_axis],
                  flag: if(axis_size <= 3, do: rem(axis_size, 2), else: 1 - rem(axis_size, 2)),
                  dx: nil
                )

            {results, val}
          else
            {results, val}
          end

        {results, val} =
          if avg or last do
            y_slice0 = Nx.take(y, 0, axis: axis)
            y_slice1 = Nx.take(y, 1, axis: axis)

            x_slice0 = Nx.take(x, 0, axis: axis)
            x_slice1 = Nx.take(x, 1, axis: axis)

            first_dx = x_slice1 - x_slice0

            val = val + 0.5 * first_dx * (y_slice0 + y_slice1)

            results =
              results +
                basic_simpson(y, 1, x,
                  len: if(axis_size - 3 >= 0, do: axis_size - 3, else: 3 - axis_size),
                  axis: axis,
                  keep_axis: opts[:keep_axis],
                  flag: if(axis_size <= 3, do: rem(axis_size, 2), else: 1 - rem(axis_size, 2)),
                  dx: nil
                )

            {results, val}
          else
            {results, val}
          end

        {results, val} =
          if avg do
            {results / 2.0, val / 2.0}
          else
            {results, val}
          end

        results + val

      true ->
        basic_simpson(y, 0, x,
          len: axis_size - 2,
          axis: axis,
          keep_axis: opts[:keep_axis],
          flag: if(axis_size <= 3, do: rem(axis_size, 2), else: 1 - rem(axis_size, 2)),
          dx: nil
        )
    end
  end

  @doc """
  Integrate `y` along the given axis using the composite trapezoidal rule.

  This is a simplified version of `trapezoidal/3` that assumes `x` is
  a uniform tensor along `axis` with step size equal to `dx`.

  ## Options

  #{NimbleOptions.docs(@uniform_simpson_schema)}

  ## Examples

      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Integrate.simpson_uniform(y)
      #Nx.Tensor<
        f32
        4.0
      >

      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Integrate.simpson_uniform(y, dx: 2)
      #Nx.Tensor<
        f32
        8.0
      >

      iex> y = Nx.tensor([[0, 1, 2], [3, 4, 5]])
      iex> Scholar.Integrate.simpson_uniform(y, dx: 2, axis: 0)
      #Nx.Tensor<
        f32[3]
        [3.0, 5.0, 7.0]
      >
  """
  deftransform simpson_uniform(y, opts \\ []) do
    simpson_uniform_n(y, NimbleOptions.validate!(opts, @uniform_simpson_schema))
  end

  defnp simpson_uniform_n(y, opts \\ []) do
    axis = opts[:axis]
    axis_size = Nx.axis_size(y, axis)
    {avg, first, last} = parse_even(opts[:even])
    dx = opts[:dx]
    first_dx = dx
    last_dx = dx

    results = Nx.tensor(0.0, type: to_float_type(y))
    val = Nx.tensor(0.0, type: to_float_type(y))

    cond do
      rem(axis_size, 2) == 0 ->
        {results, val} =
          if avg or first do
            y_slice0 = Nx.take(y, axis_size - 1, axis: axis)
            y_slice1 = Nx.take(y, axis_size - 2, axis: axis)

            val = val + 0.5 * last_dx * (y_slice0 + y_slice1)

            results =
              results +
                basic_simpson(y, 0, 0,
                  len: if(axis_size - 3 >= 0, do: axis_size - 3, else: 3 - axis_size),
                  axis: axis,
                  keep_axis: opts[:keep_axis],
                  flag: if(axis_size <= 3, do: rem(axis_size, 2), else: 1 - rem(axis_size, 2)),
                  dx: dx
                )

            {results, val}
          else
            {results, val}
          end

        {results, val} =
          if avg or last do
            y_slice0 = Nx.take(y, 0, axis: axis)
            y_slice1 = Nx.take(y, 1, axis: axis)

            val = val + 0.5 * first_dx * (y_slice0 + y_slice1)

            results =
              results +
                basic_simpson(y, 1, 0,
                  len: if(axis_size - 3 >= 0, do: axis_size - 3, else: 3 - axis_size),
                  axis: axis,
                  keep_axis: opts[:keep_axis],
                  flag: if(axis_size <= 3, do: rem(axis_size, 2), else: 1 - rem(axis_size, 2)),
                  dx: dx
                )

            {results, val}
          else
            {results, val}
          end

        {results, val} =
          if avg do
            {results / 2.0, val / 2.0}
          else
            {results, val}
          end

        results + val

      true ->
        basic_simpson(y, 0, 0,
          len: axis_size - 2,
          axis: axis,
          keep_axis: opts[:keep_axis],
          flag: if(axis_size <= 3, do: rem(axis_size, 2), else: 1 - rem(axis_size, 2)),
          dx: dx
        )
    end
  end

  defnp basic_simpson(y, start, x, opts) do
    axis = opts[:axis]
    len = opts[:len]
    flag = opts[:flag]

    slice_0 =
      if start > Nx.axis_size(y, axis) - 1 or (start == Nx.axis_size(y, axis) - 1 and flag == 0),
        do: Nx.broadcast(0.0, Nx.new_axis(Nx.take(y, 0, axis: axis), axis)),
        else:
          Nx.slice_along_axis(
            y,
            start,
            len,
            axis: axis,
            strides: 2
          )

    slice_1 =
      if start + 1 > Nx.axis_size(y, axis) - 1 or
           (start + 1 == Nx.axis_size(y, axis) - 1 and flag == 0),
         do: Nx.broadcast(0.0, Nx.new_axis(Nx.take(y, 0, axis: axis), axis)),
         else:
           Nx.slice_along_axis(
             y,
             start + 1,
             len,
             axis: axis,
             strides: 2
           )

    slice_2 =
      if start + 2 > Nx.axis_size(y, axis) - 1 or
           (start + 2 == Nx.axis_size(y, axis) - 1 and flag == 0),
         do: Nx.broadcast(0.0, Nx.new_axis(Nx.take(y, 0, axis: axis), axis)),
         else:
           Nx.slice_along_axis(
             y,
             start + 2,
             len,
             axis: axis,
             strides: 2
           )

    case opts[:dx] do
      nil ->
        diff = Nx.diff(x, axis: axis)

        diff_0 =
          if start > Nx.axis_size(y, axis) - 2 or
               (start == Nx.axis_size(y, axis) - 2 and flag == 0),
             do: Nx.broadcast(0.0, Nx.new_axis(Nx.take(y, 0, axis: axis), axis)),
             else:
               Nx.slice_along_axis(
                 diff,
                 start,
                 len,
                 axis: axis,
                 strides: 2
               )

        diff_1 =
          if start + 1 > Nx.axis_size(y, axis) - 2 or
               (start + 1 == Nx.axis_size(y, axis) - 2 and flag == 0),
             do: Nx.broadcast(0.0, Nx.new_axis(Nx.take(y, 0, axis: axis), axis)),
             else:
               Nx.slice_along_axis(
                 diff,
                 start + 1,
                 len,
                 axis: axis,
                 strides: 2
               )

        diff_sum = diff_0 + diff_1
        diff_prod = diff_0 * diff_1
        diff_div = diff_0 / Nx.select(diff_1 == 0, 1, diff_1)

        temp =
          if start + 1 > Nx.axis_size(y, axis) - 2 or
               (start + 1 == Nx.axis_size(y, axis) - 2 and flag == 0),
             do: Nx.broadcast(0.0, Nx.new_axis(Nx.take(y, 0, axis: axis), axis)),
             else:
               diff_sum / 6.0 *
                 (slice_0 * (2.0 - 1.0 / Nx.select(diff_div == 0, 1, diff_div)) +
                    slice_1 * (diff_sum * diff_sum / Nx.select(diff_prod == 0, 1, diff_prod)) +
                    slice_2 * (2.0 - diff_div))

        Nx.sum(temp, axes: [axis], keep_axes: opts[:keep_axis])

      _ ->
        # We only calculate this part if all slices slice_i exists,
        # so we the same condition as in slice_2 since if slice_2 exists
        # then slice_0 and slice_1 exists as well.
        if start + 2 > Nx.axis_size(y, axis) - 1 or
             (start + 2 == Nx.axis_size(y, axis) - 1 and flag == 0),
           do: Nx.broadcast(0.0, Nx.broadcast(0.0, Nx.take(y, 0, axis: axis))),
           else:
             Nx.sum(slice_0 + 4 * slice_1 + slice_2, axes: [axis], keep_axes: opts[:keep_axis]) *
               (opts[:dx] / 3.0)
    end
  end

  @doc """
  Integrate `y` along the given axis using the composite trapezoidal rule.
  The integration happens in sequence along elements of `x`.

  ## Options

  #{NimbleOptions.docs(@trapezoidal_schema)}

  ## Examples

      iex> y = Nx.tensor([1, 2, 3])
      iex> x = Nx.tensor([4, 5, 6])
      iex> Scholar.Integrate.trapezoidal(y, x)
      #Nx.Tensor<
        f32
        4.0
      >

      iex> y = Nx.tensor([[0, 1, 2], [3, 4, 5]])
      iex> x = Nx.tensor([[1, 2, 3], [1, 2, 3]])
      iex> Scholar.Integrate.trapezoidal(y, x)
      #Nx.Tensor<
        f32[2]
        [2.0, 8.0]
      >

      iex> y = Nx.tensor([[0, 1, 2], [3, 4, 5]])
      iex> x = Nx.tensor([[1, 1, 1], [2, 2, 2]])
      iex> Scholar.Integrate.trapezoidal(y, x, axis: 0)
      #Nx.Tensor<
        f32[3]
        [1.5, 2.5, 3.5]
      >
  """
  deftransform trapezoidal(y, x, opts \\ []) do
    trapezoidal_n(y, x, NimbleOptions.validate!(opts, @trapezoidal_schema))
  end

  defnp trapezoidal_n(y, x, opts \\ []) do
    axis = opts[:axis]
    y_axis_size = Nx.axis_size(y, axis)
    check_shape(x, y, axis)

    d =
      if Nx.rank(x) == 1 do
        x_diff = Nx.diff(x)
        shape = prepare_shape(Nx.rank(y), Nx.size(x_diff), axis)
        Nx.reshape(x_diff, shape)
      else
        Nx.diff(x, axis: axis)
      end

    scaler = d / 2.0

    first_slice = Nx.slice_along_axis(y, 0, y_axis_size - 1, axis: axis) * scaler
    second_slice = Nx.slice_along_axis(y, 1, y_axis_size - 1, axis: axis) * scaler
    Nx.sum(first_slice + second_slice, axes: [axis], keep_axes: opts[:keep_axis])
  end

  @doc """
  Integrate `y` along the given axis using the composite trapezoidal rule.

  This is a simplified version of `trapezoidal/3` that assumes `x` is
  a uniform tensor along `axis` with step size equal to `dx`.

  ## Options

  #{NimbleOptions.docs(@uniform_trapezoidal_schema)}

  ## Examples

      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Integrate.trapezoidal_uniform(y)
      #Nx.Tensor<
        f32
        4.0
      >

      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Integrate.trapezoidal_uniform(y, dx: 2)
      #Nx.Tensor<
        f32
        8.0
      >

      iex> y = Nx.tensor([[0, 1, 2], [3, 4, 5]])
      iex> Scholar.Integrate.trapezoidal_uniform(y, dx: 2, axis: 0)
      #Nx.Tensor<
        f32[3]
        [3.0, 5.0, 7.0]
      >
  """
  deftransform trapezoidal_uniform(y, opts \\ []) do
    trapezoidal_uniform_n(y, NimbleOptions.validate!(opts, @uniform_trapezoidal_schema))
  end

  defnp trapezoidal_uniform_n(y, opts \\ []) do
    axis = opts[:axis]
    y_axis_size = Nx.axis_size(y, axis)

    scaler = opts[:dx] / 2.0
    y = y * scaler

    first_slice = Nx.slice_along_axis(y, 0, y_axis_size - 1, axis: axis)
    second_slice = Nx.slice_along_axis(y, 1, y_axis_size - 1, axis: axis)
    Nx.sum(first_slice + second_slice, axes: [axis], keep_axes: opts[:keep_axis])
  end

  deftransformp parse_even(even) do
    case even do
      :avg ->
        {1, 0, 0}

      :first ->
        {0, 1, 0}

      :last ->
        {0, 0, 1}
    end
  end

  deftransformp check_shape(x, y, axis) do
    x_shape = Nx.shape(x)
    y_shape = Nx.shape(y)
    x_rank = Nx.rank(x)
    y_rank = Nx.rank(y)
    y_axis_size = Nx.axis_size(y, axis)

    if x_rank == 1 and Nx.size(x) != y_axis_size do
      raise ArgumentError, "x and y must have the same size along the given axis"
    end

    if x_rank != 1 do
      x_axis_size = Nx.axis_size(x, axis)

      cond do
        x_rank != y_rank ->
          raise ArgumentError, "x must be rank 1 or x and y must have the same rank"

        x_axis_size != y_axis_size ->
          raise ArgumentError, "x and y must have the same size along the given axis"

        true ->
          nil
      end

      valid_broadcast!(x_rank, x_shape, y_shape)
    end
  end

  deftransformp prepare_shape(y_rank, x_size, axis) do
    List.duplicate(1, y_rank)
    |> List.replace_at(axis, x_size)
    |> List.to_tuple()
  end
end
