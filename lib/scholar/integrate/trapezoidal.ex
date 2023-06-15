defmodule Scholar.Integrate.Trapezoidal do
  @moduledoc """
  Univariate imputer for completing missing values with simple strategies.
  """
  import Nx.Defn

  general = [
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
      doc: "If set to true, the axes which are reduced are left."
    ]
  ]

  uniform_schema =
    general ++
      [
        dx: [
          type: {:or, [:float, :integer]},
          default: 1.0,
          doc: """
          The spacing between samples.
          """
        ]
      ]

  @trapezoidal_schema NimbleOptions.new!(general)
  @uniform_schema NimbleOptions.new!(uniform_schema)

  @doc """
  Integrate `y` along the given axis using the composite trapezoidal rule.
  The integration happens in sequence along elements of `x`.

  ## Options

  #{NimbleOptions.docs(@trapezoidal_schema)}

  ## Examples

      iex> Scholar.Integrate.Trapezoidal.trapezoidal(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))
      #Nx.Tensor<
        f32
        4.0
      >

      iex> Scholar.Integrate.Trapezoidal.trapezoidal(Nx.tensor([[0, 1, 2], [3, 4, 5]]), Nx.tensor([[1, 2, 3], [1, 2, 3]]))
      #Nx.Tensor<
        f32[2]
        [2.0, 8.0]
      >

      iex> Scholar.Integrate.Trapezoidal.trapezoidal(Nx.tensor([[0, 1, 2], [3, 4, 5]]), Nx.tensor([[1, 1, 1], [2, 2, 2]]), axis: 0)
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
        shape = prepare_shape(Nx.rank(y), Nx.shape(x), axis)
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

  #{NimbleOptions.docs(@uniform_schema)}

  ## Examples

      iex> Scholar.Integrate.Trapezoidal.trapezoidal_uniform(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f32
        4.0
      >

      iex> Scholar.Integrate.Trapezoidal.trapezoidal_uniform(Nx.tensor([1, 2, 3]), dx: 2)
      #Nx.Tensor<
        f32
        8.0
      >

      iex> Scholar.Integrate.Trapezoidal.trapezoidal_uniform(Nx.tensor([[0, 1, 2], [3, 4, 5]]), dx: 2, axis: 0)
      #Nx.Tensor<
        f32[3]
        [3.0, 5.0, 7.0]
      >
  """
  deftransform trapezoidal_uniform(y, opts \\ []) do
    trapezoidal_uniform_n(y, NimbleOptions.validate!(opts, @uniform_schema))
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

        not valid_broadcast?(Enum.to_list(0..(x_rank - 1)), x_shape, y_shape) ->
          raise ArgumentError,
                "x and y must be broadcast compatible with dimension #{inspect(axis)} of same size"

        true ->
          nil
      end
    end
  end

  deftransform prepare_shape(y_rank, x_shape, axis) do
    List.duplicate(1, y_rank)
    |> List.replace_at(axis, elem(x_shape, 0) - 1)
    |> List.to_tuple()
  end

  defp valid_broadcast?([head | tail], old_shape, new_shape) do
    old_dim = elem(old_shape, head)
    new_dim = elem(new_shape, head)

    (old_dim == 1 or old_dim == new_dim) and
      valid_broadcast?(tail, old_shape, new_shape)
  end

  defp valid_broadcast?([], _old_shape, _new_shape), do: true
end
