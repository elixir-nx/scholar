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

    * `:axes` - Axes to aggregate distance over. If `:axes` set to `nil` then function does not aggregate distances.
      Defaults to `nil`.

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
      ** (ArgumentError) expected input shapes to be equal, got {2} != {3}

      iex> x = Nx.tensor([[1,2], [3,4]])
      iex> y = Nx.tensor([[8,3], [2,5]])
      iex> Scholar.Metrics.Distance.euclidean(x, y, axes: [0])
      #Nx.Tensor<
        f32[2]
        [7.071067810058594, 1.4142135381698608]
      >
  """
  @spec euclidean(Nx.t(), Nx.t()) :: Nx.t()
  defn euclidean(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts =
      keyword!(
        opts,
        axes: nil
      )

    diff = x - y

    if Nx.all(diff == 0) do
      0.0
    else
      Nx.LinAlg.norm(diff, axes: opts[:axes])
    end
  end

  @doc """
  Squared euclidean distance.

  $$
  D(x, y) = \\sum_i (x_i - y_i)^2
  $$

  ## Options

  * `:axes` - Axes to aggregate distance over. If `:axes` set to `nil` then function does not aggregate distances.
    Defaults to `nil`.

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
      ** (ArgumentError) expected input shapes to be equal, got {2} != {3}

      iex> x = Nx.tensor([[1,2], [3,4]])
      iex> y = Nx.tensor([[8,3], [2,5]])
      iex> Scholar.Metrics.Distance.squared_euclidean(x, y, axes: [0])
      #Nx.Tensor<
        f32[2]
        [50.0, 2.0]
      >
  """
  @spec squared_euclidean(Nx.t(), Nx.t()) :: Nx.t()
  defn squared_euclidean(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts =
      keyword!(
        opts,
        axes: nil
      )

    x
    |> Nx.subtract(y)
    |> Nx.power(2)
    |> Nx.sum(axes: opts[:axes])
    |> as_float()
  end

  @doc """
  Manhattan, taxicab, or l1 distance.

  $$
  D(x, y) = \\sum_i |x_i - y_i|
  $$

  ## Options

  * `:axes` - Axes to aggregate distance over. If `:axes` set to `nil` then function does not aggregate distances.
    Defaults to `nil`.

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
      ** (ArgumentError) expected input shapes to be equal, got {2} != {3}

      iex> x = Nx.tensor([[1,2], [3,4]])
      iex> y = Nx.tensor([[8,3], [2,5]])
      iex> Scholar.Metrics.Distance.manhattan(x, y, axes: [0])
      #Nx.Tensor<
        f32[2]
        [8.0, 2.0]
      >
  """
  @spec manhattan(Nx.t(), Nx.t()) :: Nx.t()
  defn manhattan(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts =
      keyword!(
        opts,
        axes: nil
      )

    x
    |> Nx.subtract(y)
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

  * `:axes` - Axes to aggregate distance over. If `:axes` set to `nil` then function does not aggregate distances.
    Defaults to `nil`.

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
      ** (ArgumentError) expected input shapes to be equal, got {2} != {3}

      iex> x = Nx.tensor([[1,2], [3,4]])
      iex> y = Nx.tensor([[8,3], [2,5]])
      iex> Scholar.Metrics.Distance.chebyshev(x, y, axes: [1])
      #Nx.Tensor<
        f32[2]
        [7.0, 1.0]
      >
  """
  @spec chebyshev(Nx.t(), Nx.t()) :: Nx.t()
  defn chebyshev(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts =
      keyword!(
        opts,
        axes: nil
      )

    x
    |> Nx.subtract(y)
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

  * `:axes` - Axes to aggregate distance over. If `:axes` set to `nil` then function does not aggregate distances.
    Defaults to `nil`.

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
      ** (ArgumentError) expected input shapes to be equal, got {2} != {3}

      iex> x = Nx.tensor([[1,2], [3,4]])
      iex> y = Nx.tensor([[8,3], [2,5]])
      iex> Scholar.Metrics.Distance.minkowski(x, y, p: 2.5, axes: [0])
      #Nx.Tensor<
        f32[2]
        [7.021548271179199, 1.3195079565048218]
      >
  """
  @spec minkowski(Nx.t(), Nx.t()) :: Nx.t()
  defn minkowski(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts =
      keyword!(
        opts,
        p: 2,
        axes: nil
      )

    p = Nx.tensor(opts[:p])
    check_p(p)

    case p do
      0 ->
        chebyshev(x, y, axes: opts[:axes])

      1 ->
        manhattan(x, y, axes: opts[:axes])

      2 ->
        euclidean(x, y, axes: opts[:axes])

      _ ->
        x
        |> Nx.subtract(y)
        |> Nx.abs()
        |> Nx.power(p)
        |> Nx.sum(axes: opts[:axes])
        |> Nx.power(1.0 / p)
    end
  end

  deftransformp check_p(p) do
    if p < 0 do
      raise ArgumentError,
            "The value of p must be non-negative"
    end
  end

  @doc """
  Cosine distance. It only accepts 2D tensors.

  $$
  1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}
  $$

  ## Options

  * `:axes` - Axes to aggregate distance over. If `:axes` set to `nil` then function does not aggregate distances.
    Defaults to `nil`.

  ## Examples

      iex> x = Nx.tensor([[1, 2]])
      iex> y = Nx.tensor([[5, 2]])
      iex> Scholar.Metrics.Distance.cosine(x, y)
      #Nx.Tensor<
        f32[1]
        [0.25259071588516235]
      >

      iex> x = Nx.tensor([[1, 2]])
      iex> y = Nx.tensor([[1, 2]])
      iex> Scholar.Metrics.Distance.cosine(x, y)
      #Nx.Tensor<
        f32[1]
        [0.0]
      >

      iex> x = Nx.tensor([[1, 2]])
      iex> y = Nx.tensor([[1, 2, 3]])
      iex> Scholar.Metrics.Distance.cosine(x, y)
      ** (ArgumentError) expected input shapes to be equal, got {1, 2} != {1, 3}

      iex> x = Nx.tensor([[1, 2, 3], [0, 0, 0], [5, 2, 4]])
      iex> y = Nx.tensor([[1, 5, 2], [2, 4, 1], [0, 0, 0]])
      iex> Scholar.Metrics.Distance.cosine(x, y, axes: [1])
      #Nx.Tensor<
        f32[3][3]
        [
          [0.1704850196838379, 0.2418246865272522, 1.0],
          [1.0, 1.0, 0.0],
          [0.3740193247795105, 0.2843400239944458, 1.0]
        ]
      >
  """
  @spec cosine(Nx.t(), Nx.t()) :: Nx.t()
  defn cosine(x, y, opts \\ []) do
    cutoff = 10 * 2.220446049250313e-16
    assert_same_shape!(x, y)

    {m, n} = Nx.shape(x)

    opts =
      keyword!(
        opts,
        axes: nil
      )

    norm_x = Nx.LinAlg.norm(x, axes: opts[:axes]) |> Nx.reshape({m, 1}) |> Nx.broadcast({m, n})
    norm_y = Nx.LinAlg.norm(y, axes: opts[:axes]) |> Nx.reshape({1, m}) |> Nx.broadcast({n, m})

    zero_mask = norm_x == 0.0 and norm_y == 0.0
    zero_xor_one_mask = Nx.logical_xor(norm_x == 0.0, norm_y == 0.0)

    norm_y = Nx.transpose(norm_y)

    norm_x = Nx.select(Nx.greater(norm_x, cutoff), norm_x, 1.0)
    norm_y = Nx.select(Nx.greater(norm_y, cutoff), norm_y, 1.0)

    norm_x = x / norm_x
    norm_y = y / norm_y

    res = Nx.dot(norm_x, Nx.transpose(norm_y))
    res = Nx.select(zero_xor_one_mask, 0.0, res)
    res = 1.0 - Nx.select(zero_mask, 1.0, res)
    if m != 1, do: res, else: Nx.new_axis(res[0][0], 0)
  end

  defnp as_float(x) do
    transform(x, fn x ->
      x_f = Nx.Type.to_floating(x.type)
      Nx.as_type(x, x_f)
    end)
  end
end
