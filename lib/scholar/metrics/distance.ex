defmodule Scholar.Metrics.Distance do
  @moduledoc """
  Distance metrics between multi-dimensional tensors.
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

  minkowski_schema =
    general_schema ++
      [
        p: [
          type: {:or, [{:custom, Scholar.Options, :positive_number, []}, {:in, [:infinity]}]},
          default: 2.0,
          doc: """
          A positive parameter of Minkowski distance or :infinity (then Chebyshev metric computed).
          """
        ]
      ]

  @general_schema NimbleOptions.new!(general_schema)
  @minkowski_schema NimbleOptions.new!(minkowski_schema)

  @doc """
  Standard euclidean distance ($L_{2}$ distance).

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

  defnp euclidean_n(x, y, opts) do
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

  defnp squared_euclidean_n(x, y, opts) do
    assert_same_shape!(x, y)

    diff = x - y

    (diff * diff)
    |> Nx.sum(axes: opts[:axes])
    |> to_float()
  end

  @doc """
  Manhattan, Taxicab, or $L_{1}$ distance.

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

  defnp manhattan_n(x, y, opts) do
    assert_same_shape!(x, y)

    (x - y)
    |> Nx.abs()
    |> Nx.sum(axes: opts[:axes])
    |> to_float()
  end

  @doc """
  Chebyshev or $L_{\\infty}$ distance.

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

  defnp chebyshev_n(x, y, opts) do
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

  defnp minkowski_n(x, y, opts) do
    assert_same_shape!(x, y)

    p = opts[:p]

    cond do
      p == :infinity ->
        chebyshev(x, y, axes: opts[:axes])

      p == 1 ->
        manhattan(x, y, axes: opts[:axes])

      p == 2 ->
        euclidean(x, y, axes: opts[:axes])

      true ->
        (x - y)
        |> Nx.abs()
        |> Nx.pow(p)
        |> Nx.sum(axes: opts[:axes])
        |> Nx.pow(1.0 / p)
    end
  end

  @doc """
  Cosine distance.

  $$
  D(u, v) = 1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}
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

  defnp cosine_n(x, y, opts) do
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

    norm_x = Nx.select(norm_x > cutoff, norm_x, Nx.tensor(1.0, type: to_float_type(x)))
    normalized_x = x / norm_x

    norm_y = Nx.select(norm_y > cutoff, norm_y, Nx.tensor(1.0, type: to_float_type(y)))
    normalized_y = y / norm_y

    norm_x = Nx.squeeze(norm_x, axes: opts[:axes])
    norm_y = Nx.squeeze(norm_y, axes: opts[:axes])

    x_zero? = norm_x == 0.0
    y_zero? = norm_y == 0.0

    both_zero? = x_zero? and y_zero?
    one_zero? = Nx.logical_xor(x_zero?, y_zero?)

    res = (normalized_x * normalized_y) |> Nx.sum(axes: opts[:axes])
    merged_type = Nx.Type.merge(Nx.type(x), Nx.type(y))
    res = Nx.select(one_zero?, Nx.tensor(0, type: merged_type), res)
    one_merged_type = Nx.tensor(1, type: merged_type)
    one_merged_type - Nx.select(both_zero?, one_merged_type, res)
  end

  @doc """
  Hamming distance.

  #{~S'''
  $$
  hamming(x, y) = \frac{\left \lvert x\_{i, j, ...} \neq y\_{i, j, ...}\right \rvert}{\left \lvert x\_{i, j, ...}\right \rvert}
  $$
  where $i, j, ...$ are the aggregation axes
  '''}

  ## Options

  #{NimbleOptions.docs(@general_schema)}

  ## Examples

      iex> x = Nx.tensor([1, 0, 0])
      iex> y = Nx.tensor([0, 1, 0])
      iex> Scholar.Metrics.Distance.hamming(x, y)
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Distance.hamming(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}

      iex> x = Nx.tensor([[1, 2, 3], [0, 0, 0], [5, 2, 4]])
      iex> y = Nx.tensor([[1, 5, 2], [2, 4, 1], [0, 0, 0]])
      iex> Scholar.Metrics.Distance.hamming(x, y, axes: [1])
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 1.0, 1.0]
      >
  """
  deftransform hamming(x, y, opts \\ [])

  deftransform hamming(x, y, opts) when is_list(opts) do
    NimbleOptions.validate!(opts, @general_schema)
    hamming_unweighted(x, y, opts)
  end

  deftransform hamming(x, y, w), do: hamming_weighted(x, y, w, [])

  @doc """
  Hamming distance in weighted version.

  ## Options

  #{NimbleOptions.docs(@general_schema)}

  ## Examples

      iex> x = Nx.tensor([1, 0, 0])
      iex> y = Nx.tensor([0, 1, 0])
      iex> weights = Nx.tensor([1, 0.5, 0.5])
      iex> Scholar.Metrics.Distance.hamming(x, y, weights)
      #Nx.Tensor<
        f32
        0.75
      >
  """
  deftransform hamming(x, y, w, opts) when is_list(opts) do
    NimbleOptions.validate!(opts, @general_schema)
    hamming_weighted(x, y, w, opts)
  end

  defnp hamming_unweighted(x, y, opts) do
    assert_same_shape!(x, y)
    result_type = Nx.Type.to_floating(Nx.Type.merge(Nx.type(x), Nx.type(y)))
    Nx.mean(x != y, axes: opts[:axes]) |> Nx.as_type(result_type)
  end

  defnp hamming_weighted(x, y, w, opts) do
    assert_same_shape!(x, y)
    result_type = Nx.Type.to_floating(Nx.Type.merge(Nx.type(x), Nx.type(y)))
    w = Nx.as_type(w, result_type)
    Nx.weighted_mean(x != y, w, axes: opts[:axes]) |> Nx.as_type(result_type)
  end
end
