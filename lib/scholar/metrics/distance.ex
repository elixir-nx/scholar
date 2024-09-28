defmodule Scholar.Metrics.Distance do
  @moduledoc """
  Distance metrics between multi-dimensional tensors.
  They all support distance calculations between any subset of axes.
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

  pairwise_minkowski_schema =
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
  @minkowski_schema NimbleOptions.new!(general_schema ++ pairwise_minkowski_schema)
  @pairwise_minkowski_schema NimbleOptions.new!(pairwise_minkowski_schema)

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
      ** (ArgumentError) tensors must be broadcast compatible, got tensors with shapes {2} and {3}

      iex> x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      iex> y = Nx.tensor([[8, 3, 1], [2, 5, 2]])
      iex> Scholar.Metrics.Distance.euclidean(x, y, axes: [0])
      #Nx.Tensor<
        f32[3]
        [7.071067810058594, 1.4142135381698608, 4.123105525970459]
      >

      iex> x = Nx.tensor([[6, 2, 9], [2, 5, 3]])
      iex> y = Nx.tensor([[8, 3, 1]])
      iex> Scholar.Metrics.Distance.euclidean(x, y)
      #Nx.Tensor<
        f32
        10.630146026611328
      >
  """
  deftransform euclidean(x, y, opts \\ []) do
    euclidean_n(x, y, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp euclidean_n(x, y, opts) do
    valid_broadcast!(Nx.rank(x), Nx.shape(x), Nx.shape(y))
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
      ** (ArgumentError) tensors must be broadcast compatible, got tensors with shapes {2} and {3}

      iex> x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      iex> y = Nx.tensor([[8, 3, 1], [2, 5, 2]])
      iex> Scholar.Metrics.Distance.squared_euclidean(x, y, axes: [0])
      #Nx.Tensor<
        f32[3]
        [50.0, 2.0, 17.0]
      >

      iex> x = Nx.tensor([[6, 2, 9], [2, 5, 3]])
      iex> y = Nx.tensor([[8, 3, 1]])
      iex> Scholar.Metrics.Distance.squared_euclidean(x, y)
      #Nx.Tensor<
        f32
        113.0
      >
  """
  deftransform squared_euclidean(x, y, opts \\ []) do
    squared_euclidean_n(x, y, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp squared_euclidean_n(x, y, opts) do
    valid_broadcast!(Nx.rank(x), Nx.shape(x), Nx.shape(y))
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
      ** (ArgumentError) tensors must be broadcast compatible, got tensors with shapes {2} and {3}

      iex> x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      iex> y = Nx.tensor([[8, 3, 1], [2, 5, 2]])
      iex> Scholar.Metrics.Distance.manhattan(x, y, axes: [0])
      #Nx.Tensor<
        f32[3]
        [8.0, 2.0, 5.0]
      >

      iex> x = Nx.tensor([[6, 2, 9], [2, 5, 3]])
      iex> y = Nx.tensor([[8, 3, 1]])
      iex> Scholar.Metrics.Distance.manhattan(x, y)
      #Nx.Tensor<
        f32
        21.0
      >
  """
  deftransform manhattan(x, y, opts \\ []) do
    manhattan_n(x, y, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp manhattan_n(x, y, opts) do
    valid_broadcast!(Nx.rank(x), Nx.shape(x), Nx.shape(y))

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
      ** (ArgumentError) tensors must be broadcast compatible, got tensors with shapes {2} and {3}

      iex> x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      iex> y = Nx.tensor([[8, 3, 1], [2, 5, 2]])
      iex> Scholar.Metrics.Distance.chebyshev(x, y, axes: [1])
      #Nx.Tensor<
        f32[2]
        [7.0, 1.0]
      >

      iex> x = Nx.tensor([[6, 2, 9], [2, 5, 3]])
      iex> y = Nx.tensor([[8, 3, 1]])
      iex> Scholar.Metrics.Distance.chebyshev(x, y)
      #Nx.Tensor<
        f32
        8.0
      >
  """
  deftransform chebyshev(x, y, opts \\ []) do
    chebyshev_n(x, y, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp chebyshev_n(x, y, opts) do
    valid_broadcast!(Nx.rank(x), Nx.shape(x), Nx.shape(y))

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
      ** (ArgumentError) tensors must be broadcast compatible, got tensors with shapes {2} and {3}

      iex> x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      iex> y = Nx.tensor([[8, 3, 1], [2, 5, 2]])
      iex> Scholar.Metrics.Distance.minkowski(x, y, p: 2.5, axes: [0])
      #Nx.Tensor<
        f32[3]
        [7.021548271179199, 1.3195079565048218, 4.049539089202881]
      >

      iex> x = Nx.tensor([[6, 2, 9], [2, 5, 3]])
      iex> y = Nx.tensor([[8, 3, 1]])
      iex> Scholar.Metrics.Distance.minkowski(x, y, p: 2.5)
      #Nx.Tensor<
        f32
        9.621805191040039
      >
  """
  deftransform minkowski(x, y, opts \\ []) do
    minkowski_n(x, y, NimbleOptions.validate!(opts, @minkowski_schema))
  end

  defnp minkowski_n(x, y, opts) do
    valid_broadcast!(Nx.rank(x), Nx.shape(x), Nx.shape(y))
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
  Computes the pairwise Minkowski distance.

  ## Examples

      iex> x = Nx.iota({2, 3})
      iex> y = Nx.reverse(x)
      iex> Scholar.Metrics.Distance.pairwise_minkowski(x, y)
      #Nx.Tensor<
        f32[2][2]
        [
          [5.916079998016357, 2.8284270763397217],
          [2.8284270763397217, 5.916079998016357]
        ]
      >
  """
  deftransform pairwise_minkowski(x, y, opts \\ []) do
    pairwise_minkowski_n(x, y, NimbleOptions.validate!(opts, @pairwise_minkowski_schema))
  end

  defnp pairwise_minkowski_n(x, y, opts) do
    p = opts[:p]

    cond do
      p == 2 ->
        pairwise_euclidean(x, y)

      true ->
        x = Nx.new_axis(x, 1)
        y = Nx.new_axis(y, 0)
        minkowski_n(x, y, axes: [-1], p: p)
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
      ** (ArgumentError) tensors must be broadcast compatible, got tensors with shapes {2} and {3}

      iex> x = Nx.tensor([[1, 2, 3], [0, 0, 0], [5, 2, 4]])
      iex> y = Nx.tensor([[1, 5, 2], [2, 4, 1], [0, 0, 0]])
      iex> Scholar.Metrics.Distance.cosine(x, y, axes: [1])
      #Nx.Tensor<
        f32[3]
        [0.1704850196838379, 1.0, 1.0]
      >

      iex> x = Nx.tensor([[6, 2, 9], [2, 5, 3]])
      iex> y = Nx.tensor([[8, 3, 1]])
      iex> Scholar.Metrics.Distance.cosine(x, y)
      #Nx.Tensor<
        f32
        0.10575336217880249
      >
  """
  deftransform cosine(x, y, opts \\ []) do
    cosine_n(x, y, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp cosine_n(x, y, opts) do
    valid_broadcast!(Nx.rank(x), Nx.shape(x), Nx.shape(y))
    # Detect very small values that could lead to surprising
    # results and numerical stability issues. Every value smaller
    # than `cutoff` is considered small
    cutoff = 10 * Nx.Constants.epsilon(:f64)

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
    Nx.max(0, one_merged_type - Nx.select(both_zero?, one_merged_type, res))
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
      ** (ArgumentError) tensors must be broadcast compatible, got tensors with shapes {2} and {3}

      iex> x = Nx.tensor([[1, 2, 3], [0, 0, 0], [5, 2, 4]])
      iex> y = Nx.tensor([[1, 5, 2], [2, 4, 1], [0, 0, 0]])
      iex> Scholar.Metrics.Distance.hamming(x, y, axes: [1])
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 1.0, 1.0]
      >

      iex> x = Nx.tensor([[6, 2, 9], [2, 5, 3]])
      iex> y = Nx.tensor([[8, 3, 1]])
      iex> Scholar.Metrics.Distance.hamming(x, y)
      #Nx.Tensor<
        f32
        1.0
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

      iex> x = Nx.tensor([[6, 2, 9], [2, 5, 3]])
      iex> y = Nx.tensor([[8, 3, 1]])
      iex> weights = Nx.tensor([1, 0.5, 0.5])
      iex> Scholar.Metrics.Distance.hamming(x, y, weights, axes: [1])
      #Nx.Tensor<
        f32[2]
        [1.0, 1.0]
      >
  """
  deftransform hamming(x, y, w, opts) when is_list(opts) do
    NimbleOptions.validate!(opts, @general_schema)
    hamming_weighted(x, y, w, opts)
  end

  defnp hamming_unweighted(x, y, opts) do
    valid_broadcast!(Nx.rank(x), Nx.shape(x), Nx.shape(y))
    result_type = Nx.Type.to_floating(Nx.Type.merge(Nx.type(x), Nx.type(y)))
    Nx.mean(x != y, axes: opts[:axes]) |> Nx.as_type(result_type)
  end

  defnp hamming_weighted(x, y, w, opts) do
    valid_broadcast!(Nx.rank(x), Nx.shape(x), Nx.shape(y))
    result_type = Nx.Type.to_floating(Nx.Type.merge(Nx.type(x), Nx.type(y)))
    w = Nx.as_type(w, result_type)
    Nx.weighted_mean(x != y, w, axes: opts[:axes]) |> Nx.as_type(result_type)
  end

  @doc """
  Pairwise squared euclidean distance.

  ## Examples

      iex> x = Nx.iota({6, 6})
      iex> y = Nx.reverse(x)
      iex> Scholar.Metrics.Distance.pairwise_squared_euclidean(x, y)
      #Nx.Tensor<
        s64[6][6]
        [
          [5470, 3526, 2014, 934, 286, 70],
          [3526, 2014, 934, 286, 70, 286],
          [2014, 934, 286, 70, 286, 934],
          [934, 286, 70, 286, 934, 2014],
          [286, 70, 286, 934, 2014, 3526],
          [70, 286, 934, 2014, 3526, 5470]
        ]
      >
  """
  defn pairwise_squared_euclidean(x, y) do
    y_norm = Nx.sum(y * y, axes: [1]) |> Nx.new_axis(0)
    x_norm = Nx.sum(x * x, axes: [1], keep_axes: true)
    Nx.max(0, x_norm + y_norm - 2 * Nx.dot(x, [-1], y, [-1]))
  end

  @doc """
  Pairwise squared euclidean distance. It is equivalent to
  Scholar.Metrics.Distance.pairwise_squared_euclidean(x, x)

  ## Examples

      iex> x = Nx.iota({6, 6})
      iex> Scholar.Metrics.Distance.pairwise_squared_euclidean(x)
      #Nx.Tensor<
        s64[6][6]
        [
          [0, 216, 864, 1944, 3456, 5400],
          [216, 0, 216, 864, 1944, 3456],
          [864, 216, 0, 216, 864, 1944],
          [1944, 864, 216, 0, 216, 864],
          [3456, 1944, 864, 216, 0, 216],
          [5400, 3456, 1944, 864, 216, 0]
        ]
      >
  """
  defn pairwise_squared_euclidean(x) do
    x_norm = Nx.sum(x * x, axes: [1], keep_axes: true)
    dist = Nx.max(0, x_norm + Nx.transpose(x_norm) - 2 * Nx.dot(x, [-1], x, [-1]))
    Nx.put_diagonal(dist, Nx.broadcast(Nx.tensor(0, type: Nx.type(dist)), {Nx.axis_size(x, 0)}))
  end

  @doc """
  Pairwise euclidean distance.

  ## Examples

      iex> x = Nx.iota({2, 3})
      iex> y = Nx.reverse(x)
      iex> Scholar.Metrics.Distance.pairwise_euclidean(x, y)
      #Nx.Tensor<
        f32[2][2]
        [
          [5.916079998016357, 2.8284270763397217],
          [2.8284270763397217, 5.916079998016357]
        ]
      >
  """
  defn pairwise_euclidean(x, y) do
    Nx.sqrt(pairwise_squared_euclidean(x, y))
  end

  @doc """
  Pairwise euclidean distance. It is equivalent to
  Scholar.Metrics.Distance.pairwise_euclidean(x, x)

  ## Examples

      iex> x = Nx.iota({6, 6})
      iex> Scholar.Metrics.Distance.pairwise_euclidean(x)
      #Nx.Tensor<
        f32[6][6]
        [
          [0.0, 14.696938514709473, 29.393877029418945, 44.090816497802734, 58.78775405883789, 73.48469543457031],
          [14.696938514709473, 0.0, 14.696938514709473, 29.393877029418945, 44.090816497802734, 58.78775405883789],
          [29.393877029418945, 14.696938514709473, 0.0, 14.696938514709473, 29.393877029418945, 44.090816497802734],
          [44.090816497802734, 29.393877029418945, 14.696938514709473, 0.0, 14.696938514709473, 29.393877029418945],
          [58.78775405883789, 44.090816497802734, 29.393877029418945, 14.696938514709473, 0.0, 14.696938514709473],
          [73.48469543457031, 58.78775405883789, 44.090816497802734, 29.393877029418945, 14.696938514709473, 0.0]
        ]
      >
  """
  defn pairwise_euclidean(x) do
    Nx.sqrt(pairwise_squared_euclidean(x))
  end

  @doc """
  Pairwise cosine distance.

  ## Examples

      iex> x = Nx.iota({6, 6})
      iex> y = Nx.reverse(x)
      iex> Scholar.Metrics.Distance.pairwise_cosine(x, y)
      #Nx.Tensor<
        f32[6][6]
        [
          [0.2050153613090515, 0.21226388216018677, 0.22395789623260498, 0.24592703580856323, 0.30156970024108887, 0.6363636255264282],
          [0.03128105401992798, 0.03429150581359863, 0.039331674575805664, 0.049365341663360596, 0.07760530710220337, 0.30156970024108887],
          [0.014371514320373535, 0.01644366979598999, 0.020004630088806152, 0.02736520767211914, 0.049365341663360596, 0.24592703580856323],
          [0.0091819167137146, 0.010854601860046387, 0.013785064220428467, 0.020004630088806152, 0.039331674575805664, 0.22395789623260498],
          [0.006820023059844971, 0.008272230625152588, 0.010854601860046387, 0.01644366979598999, 0.03429150581359863, 0.21226388216018677],
          [0.005507469177246094, 0.006820023059844971, 0.0091819167137146, 0.014371514320373535, 0.03128105401992798, 0.2050153613090515]
        ]
      >
  """
  defn pairwise_cosine(x, y) do
    x_normalized = Scholar.Preprocessing.normalize(x, axes: [1])
    y_normalized = Scholar.Preprocessing.normalize(y, axes: [1])
    Nx.max(0, 1 - Nx.dot(x_normalized, [-1], y_normalized, [-1]))
  end

  @doc """
  Pairwise cosine distance. It is equivalent to
  Scholar.Metrics.Distance.pairwise_euclidean(x, x)

  ## Examples

      iex> x = Nx.iota({6, 6})
      iex> Scholar.Metrics.Distance.pairwise_cosine(x)
      #Nx.Tensor<
        f32[6][6]
        [
          [0.0, 0.0793418288230896, 0.1139642596244812, 0.13029760122299194, 0.1397092342376709, 0.14581435918807983],
          [0.0793418288230896, 0.0, 0.0032819509506225586, 0.006624102592468262, 0.008954286575317383, 0.01060718297958374],
          [0.1139642596244812, 0.0032819509506225586, 0.0, 5.82277774810791e-4, 0.0013980269432067871, 0.0020949840545654297],
          [0.13029760122299194, 0.006624102592468262, 5.82277774810791e-4, 0.0, 1.7595291137695312e-4, 4.686713218688965e-4],
          [0.1397092342376709, 0.008954286575317383, 0.0013980269432067871, 1.7595291137695312e-4, 0.0, 7.027387619018555e-5],
          [0.14581435918807983, 0.01060718297958374, 0.0020949840545654297, 4.686713218688965e-4, 7.027387619018555e-5, 0.0]
        ]
      >
  """
  defn pairwise_cosine(x) do
    x_normalized = Scholar.Preprocessing.normalize(x, axes: [1])
    dist = Nx.max(0, 1 - Nx.dot(x_normalized, [-1], x_normalized, [-1]))
    Nx.put_diagonal(dist, Nx.broadcast(Nx.tensor(0, type: Nx.type(dist)), {Nx.axis_size(x, 0)}))
  end
end
