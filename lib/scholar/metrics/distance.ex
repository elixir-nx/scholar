defmodule Scholar.Metrics.Distance do
  @moduledoc """
  Distance metrics between 1-D tensors.
  """

  import Nx.Defn
  import Scholar.Shared

  general_schema = [
    axes: [
      type: {:custom, Scholar.Options, :axes, []},
      doc: """
      Axes to calculate the distance over. By default the distance
      is calculated between the whole tensors.
      """
    ]
  ]

  minkowski_schema = [
    axes: [
      type: {:custom, Scholar.Options, :axes, []},
      doc: """
      Axes to calculate the distance over. By default the distance
      is calculated between the whole tensors.
      """
    ],
    p: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 2.0,
      doc: """
      A non-negative parameter of Minkowski distance.
      """
    ]
  ]

  hamming_schema = [
    axis: [
      type: :non_neg_integer,
      default: 0,
      doc: """
      Represents axis over which to compute the Hamming Distance.
      """
    ],
    weights: [
      type: {:custom, Scholar.Options, :weights, []},
      default: nil,
      doc: """
      The weights for each value in `x` and `y`. Default is nil,
      which gives each value a weight of 1.0
      """
    ]
  ]

  @general_schema NimbleOptions.new!(general_schema)
  @minkowski_schema NimbleOptions.new!(minkowski_schema)
  @hamming_schema NimbleOptions.new!(hamming_schema)

  @doc """
  Standard euclidean distance.

  $$
  D(x, y) = \\sqrt{\\sum_i (x_i - y_i)^2}
  $$

  ## Options

  #{NimbleOptions.docs(@general_schema)}

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
  deftransform euclidean(x, y, opts \\ []) do
    euclidean_n(x, y, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp euclidean_n(x, y, opts \\ []) do
    assert_same_shape!(x, y)

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

  #{NimbleOptions.docs(@general_schema)}

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
  deftransform squared_euclidean(x, y, opts \\ []) do
    squared_euclidean_n(x, y, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp squared_euclidean_n(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    diff = x - y

    (diff * diff)
    |> Nx.sum(axes: opts[:axes])
    |> to_float()
  end

  @doc """
  Manhattan, taxicab, or l1 distance.

  $$
  D(x, y) = \\sum_i |x_i - y_i|
  $$

  ## Options

  #{NimbleOptions.docs(@general_schema)}

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
  deftransform manhattan(x, y, opts \\ []) do
    manhattan_n(x, y, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp manhattan_n(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    (x - y)
    |> Nx.abs()
    |> Nx.sum(axes: opts[:axes])
    |> to_float()
  end

  @doc """
  Chebyshev or l-infinity distance.

  $$
  D(x, y) = \\max_i |x_i - y_i|
  $$

  ## Options

  #{NimbleOptions.docs(@general_schema)}

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
  deftransform chebyshev(x, y, opts \\ []) do
    chebyshev_n(x, y, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp chebyshev_n(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    (x - y)
    |> Nx.abs()
    |> Nx.reduce_max(axes: opts[:axes])
    |> to_float()
  end

  @doc """
  Minkowski distance.

  $$
  D(x, y) = \\left(\\sum_i |x_i - y_i|^p\\right)^{\\frac{1}{p}}
  $$

  ## Options

  #{NimbleOptions.docs(@minkowski_schema)}

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
  deftransform minkowski(x, y, opts \\ []) do
    minkowski_n(x, y, NimbleOptions.validate!(opts, @minkowski_schema))
  end

  defnp minkowski_n(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    p = opts[:p]

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

  #{NimbleOptions.docs(@general_schema)}

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
  deftransform cosine(x, y, opts \\ []) do
    cosine_n(x, y, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp cosine_n(x, y, opts \\ []) do
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

  @doc """
  Hamming distance.

  N_unequal(x, y) / N_tot

  ## Options

  #{NimbleOptions.docs(@hamming_schema)}

  ## Examples

      iex> x = Nx.tensor([1, 0, 0])
      iex> y = Nx.tensor([0, 1, 0])
      iex> Scholar.Metrics.Distance.hamming(x, y)
      #Nx.Tensor<
        f32
        0.6666666865348816
      >
      iex> Scholar.Metrics.Distance.hamming(x, y, weights: [1,0.5,0.5])
      #Nx.Tensor<
        f32
        0.75
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Distance.hamming(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}

      iex> x = Nx.tensor([[1, 2, 3], [0, 0, 0], [5, 2, 4]])
      iex> y = Nx.tensor([[1, 5, 2], [2, 4, 1], [0, 0, 0]])
      iex> Scholar.Metrics.Distance.hamming(x, y, axis: 1)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 1.0, 1.0]
      >
  """
  deftransform hamming(x, y, opts \\ []) do
    options = NimbleOptions.validate!(opts, @hamming_schema)

    if nil == options[:weights] do
      hamming_unweighted(x, y, options)
    else
      hamming_weighted(x, y, options)
    end
  end

  defnp hamming_unweighted(x, y, opts \\ []) do
    assert_same_shape!(x, y)
    (x != y) |> Nx.mean(axes: [opts[:axis]])
  end

  defnp hamming_weighted(x, y, opts \\ []) do
    assert_same_shape!(x, y)
    (x != y) |> Nx.weighted_mean(opts[:weights], axis: opts[:axis])
  end
end
