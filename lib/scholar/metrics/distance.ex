defmodule Scholar.Metrics.Distance do
  @moduledoc """
  Distance metrics between 1-D tensors.
  """

  import Nx.Defn
  import Scholar.Shared

  @doc """
  Standard euclidean distance.

  $$
  D(x, y) = \\sqrt{\\sum_i (x_i - y_i)^2}
  $$

  ## Options

  * `:axes` - Axes to calculate the distance over. By default the distance
    is calculated between the whole tensors.

  ## Examples

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([3, 2])
      iex> Scholar.Metrics.Distance.euclidean(x, y)
      #Nx.Tensor<
        f32
        2.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2])
      iex> Scholar.Metrics.Distance.euclidean(x, y)
      #Nx.Tensor<
        f32
        0.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Distance.euclidean(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}

      iex> x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      iex> y = Nx.tensor([[8, 3, 1], [2, 5, 2]])
      iex> Scholar.Metrics.Distance.euclidean(x, y, axes: [0])
      #Nx.Tensor<
        f32[3]
        [7.071067810058594, 1.4142135381698608, 4.123105525970459]
      >
  """
  @spec euclidean(Nx.t(), Nx.t(), keyword()) :: Nx.t()
  defn euclidean(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts = keyword!(opts, [:axes])

    diff = x - y

    (diff * diff)
    |> Nx.sum(axes: opts[:axes])
    |> Nx.sqrt()
  end

  @doc """
  Squared euclidean distance.

  $$
  D(x, y) = \\sum_i (x_i - y_i)^2
  $$

  ## Options

  * `:axes` - Axes to calculate the distance over. By default the distance
    is calculated between the whole tensors.

  ## Examples

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([3, 2])
      iex> Scholar.Metrics.Distance.squared_euclidean(x, y)
      #Nx.Tensor<
        f32
        4.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1.0, 2.0])
      iex> Scholar.Metrics.Distance.squared_euclidean(x, y)
      #Nx.Tensor<
        f32
        0.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Distance.squared_euclidean(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}

      iex> x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      iex> y = Nx.tensor([[8, 3, 1], [2, 5, 2]])
      iex> Scholar.Metrics.Distance.squared_euclidean(x, y, axes: [0])
      #Nx.Tensor<
        f32[3]
        [50.0, 2.0, 17.0]
      >
  """
  @spec squared_euclidean(Nx.t(), Nx.t(), keyword()) :: Nx.t()
  defn squared_euclidean(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts = keyword!(opts, [:axes])

    diff = x - y

    (diff * diff)
    |> Nx.sum(axes: opts[:axes])
    |> as_float()
  end

  @doc """
  Manhattan, taxicab, or l1 distance.

  $$
  D(x, y) = \\sum_i |x_i - y_i|
  $$

  ## Options

  * `:axes` - Axes to calculate the distance over. By default the distance
    is calculated between the whole tensors.

  ## Examples

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([3, 2])
      iex> Scholar.Metrics.Distance.manhattan(x, y)
      #Nx.Tensor<
        f32
        2.0
      >

      iex> x = Nx.tensor([1.0, 2.0])
      iex> y = Nx.tensor([1, 2])
      iex> Scholar.Metrics.Distance.manhattan(x, y)
      #Nx.Tensor<
        f32
        0.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Distance.manhattan(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}

      iex> x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      iex> y = Nx.tensor([[8, 3, 1], [2, 5, 2]])
      iex> Scholar.Metrics.Distance.manhattan(x, y, axes: [0])
      #Nx.Tensor<
        f32[3]
        [8.0, 2.0, 5.0]
      >
  """
  @spec manhattan(Nx.t(), Nx.t(), keyword()) :: Nx.t()
  defn manhattan(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts = keyword!(opts, [:axes])

    (x - y)
    |> Nx.abs()
    |> Nx.sum(axes: opts[:axes])
    |> as_float()
  end

  @doc """
  Chebyshev or l-infinity distance.

  $$
  D(x, y) = \\max_i |x_i - y_i|
  $$

  ## Options

  * `:axes` - Axes to calculate the distance over. By default the distance
    is calculated between the whole tensors.

  ## Examples

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([3, 2])
      iex> Scholar.Metrics.Distance.chebyshev(x, y)
      #Nx.Tensor<
        f32
        2.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2])
      iex> Scholar.Metrics.Distance.chebyshev(x, y)
      #Nx.Tensor<
        f32
        0.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Distance.chebyshev(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}

      iex> x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      iex> y = Nx.tensor([[8, 3, 1], [2, 5, 2]])
      iex> Scholar.Metrics.Distance.chebyshev(x, y, axes: [1])
      #Nx.Tensor<
        f32[2]
        [7.0, 1.0]
      >
  """
  @spec chebyshev(Nx.t(), Nx.t(), keyword()) :: Nx.t()
  defn chebyshev(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts = keyword!(opts, [:axes])

    (x - y)
    |> Nx.abs()
    |> Nx.reduce_max(axes: opts[:axes])
    |> as_float()
  end

  @doc """
  Minkowski distance.

  $$
  D(x, y) = \\left(\\sum_i |x_i - y_i|^p\\right)^{\\frac{1}{p}}
  $$

  ## Options

  * `:axes` - Axes to calculate the distance over. By default the distance
    is calculated between the whole tensors.

  * `:p` - A non-negative parameter of Minkowski distance. Defaults to `2`.

  ## Examples

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([5, 2])
      iex> Scholar.Metrics.Distance.minkowski(x, y)
      #Nx.Tensor<
        f32
        4.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2])
      iex> Scholar.Metrics.Distance.minkowski(x, y)
      #Nx.Tensor<
        f32
        0.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Distance.minkowski(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}

      iex> x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      iex> y = Nx.tensor([[8, 3, 1], [2, 5, 2]])
      iex> Scholar.Metrics.Distance.minkowski(x, y, p: 2.5, axes: [0])
      #Nx.Tensor<
        f32[3]
        [7.021548271179199, 1.3195079565048218, 4.049539089202881]
      >
  """
  @spec minkowski(Nx.t(), Nx.t(), keyword()) :: Nx.t()
  defn minkowski(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts =
      keyword!(
        opts,
        p: 2,
        axes: nil
      )

    p = opts[:p]

    if p < 0 do
      raise ArgumentError,
            "expected the value of :p to be a non-negative number, got: #{inspect(p)}"
    end

    cond do
      p == 0 ->
        chebyshev(x, y, axes: opts[:axes])

      p == 1 ->
        manhattan(x, y, axes: opts[:axes])

      p == 2 ->
        euclidean(x, y, axes: opts[:axes])

      true ->
        (x - y)
        |> Nx.abs()
        |> Nx.power(p)
        |> Nx.sum(axes: opts[:axes])
        |> Nx.power(1.0 / p)
    end
  end

  @doc """
  Cosine distance.

  $$
  1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}
  $$

  ## Options

    * `:axes` - Axes to calculate the distance over. By default the distance
      is calculated between the whole tensors.

  ## Examples

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([5, 2])
      iex> Scholar.Metrics.Distance.cosine(x, y)
      #Nx.Tensor<
        f32
        0.25259071588516235
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Distance.cosine(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}

      iex> x = Nx.tensor([[1, 2, 3], [0, 0, 0], [5, 2, 4]])
      iex> y = Nx.tensor([[1, 5, 2], [2, 4, 1], [0, 0, 0]])
      iex> Scholar.Metrics.Distance.cosine(x, y, axes: [1])
      #Nx.Tensor<
        f32[3]
        [0.1704850196838379, 1.0, 1.0]
      >
  """
  @spec cosine(Nx.t(), Nx.t(), keyword()) :: Nx.t()
  defn cosine(x, y, opts \\ []) do
    # Detect very small values that could lead to surprising
    # results and numerical stability issues. Every value smaller
    # than `cutoff` is considered small
    cutoff = 10 * 2.220446049250313e-16
    assert_same_shape!(x, y)

    opts = keyword!(opts, [:axes])

    x_squared = x * x
    y_squared = y * y

    norm_x =
      x_squared
      |> Nx.sum(axes: opts[:axes], keep_axes: true)
      |> Nx.sqrt()

    norm_y =
      y_squared
      |> Nx.sum(axes: opts[:axes], keep_axes: true)
      |> Nx.sqrt()

    norm_x = Nx.select(norm_x > cutoff, norm_x, 1.0)
    normalized_x = x / norm_x

    norm_y = Nx.select(norm_y > cutoff, norm_y, 1.0)
    normalized_y = y / norm_y

    norm_x = Nx.squeeze(norm_x, axes: opts[:axes])
    norm_y = Nx.squeeze(norm_y, axes: opts[:axes])

    x_zero? = norm_x == 0.0
    y_zero? = norm_y == 0.0

    both_zero? = x_zero? and y_zero?
    one_zero? = Nx.logical_xor(x_zero?, y_zero?)

    res = (normalized_x * normalized_y) |> Nx.sum(axes: opts[:axes])
    res = Nx.select(one_zero?, 0.0, res)
    1.0 - Nx.select(both_zero?, 1.0, res)
  end

  defnp as_float(x) do
    transform(x, fn x ->
      x_f = Nx.Type.to_floating(x.type)
      Nx.as_type(x, x_f)
    end)
  end
end
