defmodule Scholar.Preprocessing do
  @moduledoc """
  Set of functions for preprocessing data.
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

  encode_schema = [
    num_classes: [
      required: true,
      type: :pos_integer,
      doc: """
      Number of classes to be encoded.
      """
    ]
  ]

  min_max_schema =
    general_schema ++
      [
        min: [
          type: {:or, [:integer, :float]},
          default: 0,
          doc: """
          The lower boundary of the desired range of transformed data.
          """
        ],
        max: [
          type: {:or, [:integer, :float]},
          default: 1,
          doc: """
          The upper boundary of the desired range of transformed data.
          """
        ]
      ]

  normalize_schema =
    general_schema ++
      [
        norm: [
          type: {:in, [:euclidean, :chebyshev, :manhattan]},
          default: :euclidean,
          doc: """
          The norm to use to normalize each non zero sample.
          Possible options are `:euclidean`, `:manhattan`, and `:chebyshev`
          """
        ]
      ]

  binarize_schema = [
    type: [
      type: {:custom, Scholar.Options, :type, []},
      default: :f32,
      doc: """
      Type of the resultant tensor.
      """
    ],
    threshold: [
      type: {:or, [:integer, :float]},
      default: 0,
      doc: """
      Feature values below or equal to this are replaced by 0, above it by 1.
      """
    ]
  ]

  @general_schema NimbleOptions.new!(general_schema)
  @min_max_schema NimbleOptions.new!(min_max_schema)
  @normalize_schema NimbleOptions.new!(normalize_schema)
  @binarize_schema NimbleOptions.new!(binarize_schema)
  @encode_schema NimbleOptions.new!(encode_schema)

  @doc """
  Standardizes the tensor by removing the mean and scaling to unit variance.

  #{~S'''
  Formula for input tensor $x$:
  $$
  z = \frac{x - \mu}{\sigma}
  $$
  Where $\mu$ is the mean of the samples, and $\sigma$ is the standard deviation.
  Standardization can be helpful in cases where the data follows
  a Gaussian distribution (or Normal distribution) without outliers.
  '''}

  ## Options

  #{NimbleOptions.docs(@general_schema)}

  ## Examples

      iex> Scholar.Preprocessing.standard_scale(Nx.tensor([1,2,3]))
      #Nx.Tensor<
        f32[3]
        [-1.2247447967529297, 0.0, 1.2247447967529297]
      >

      iex> Scholar.Preprocessing.standard_scale(Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]]))
      #Nx.Tensor<
        f32[3][3]
        [
          [0.5212860703468323, -1.3553436994552612, 1.4596009254455566],
          [1.4596009254455566, -0.4170288145542145, -0.4170288145542145],
          [-0.4170288145542145, 0.5212860703468323, -1.3553436994552612]
        ]
      >

      iex> Scholar.Preprocessing.standard_scale(Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]]), axes: [1])
      #Nx.Tensor<
        f32[3][3]
        [
          [0.26726120710372925, -1.3363062143325806, 1.069044828414917],
          [1.4142135381698608, -0.7071068286895752, -0.7071068286895752],
          [0.0, 1.2247447967529297, -1.2247447967529297]
        ]
      >

      iex> Scholar.Preprocessing.standard_scale(42)
      #Nx.Tensor<
        f32
        42.0
      >
  """
  deftransform standard_scale(tensor, opts \\ []) do
    standard_scale_n(tensor, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp standard_scale_n(tensor, opts) do
    std = Nx.standard_deviation(tensor, axes: opts[:axes], keep_axes: true)
    mean_reduced = Nx.mean(tensor, axes: opts[:axes], keep_axes: true)
    mean_reduced = Nx.select(std == 0, 0.0, mean_reduced)
    (tensor - mean_reduced) / Nx.select(std == 0, 1.0, std)
  end

  @doc """
  Scales a tensor by dividing each sample in batch by maximum absolute value in the batch

  ## Options

  #{NimbleOptions.docs(@general_schema)}

  ## Examples

      iex> Scholar.Preprocessing.max_abs_scale(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f32[3]
        [0.3333333432674408, 0.6666666865348816, 1.0]
      >

      iex> Scholar.Preprocessing.max_abs_scale(Nx.tensor([[1, -1, 2], [3, 0, 0], [0, 1, -1], [2, 3, 1]]), axes: [0])
      #Nx.Tensor<
        f32[4][3]
        [
          [0.3333333432674408, -0.3333333432674408, 1.0],
          [1.0, 0.0, 0.0],
          [0.0, 0.3333333432674408, -0.5],
          [0.6666666865348816, 1.0, 0.5]
        ]
      >

      iex> Scholar.Preprocessing.max_abs_scale(42)
      #Nx.Tensor<
        f32
        1.0
      >
  """
  deftransform max_abs_scale(tensor, opts \\ []) do
    max_abs_scale_n(tensor, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp max_abs_scale_n(tensor, opts) do
    max_abs = Nx.abs(tensor) |> Nx.reduce_max(axes: opts[:axes], keep_axes: true)
    tensor / Nx.select(max_abs == 0, 1, max_abs)
  end

  @doc """
  Transform a tensor by scaling each batch to the given range.

  ## Options

  #{NimbleOptions.docs(@min_max_schema)}

  ## Examples

      iex> Scholar.Preprocessing.min_max_scale(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f32[3]
        [0.0, 0.5, 1.0]
      >

      iex> Scholar.Preprocessing.min_max_scale(Nx.tensor([[1, -1, 2], [3, 0, 0], [0, 1, -1], [2, 3, 1]]), axes: [0])
      #Nx.Tensor<
        f32[4][3]
        [
          [0.3333333432674408, 0.0, 1.0],
          [1.0, 0.25, 0.3333333432674408],
          [0.0, 0.5, 0.0],
          [0.6666666865348816, 1.0, 0.6666666865348816]
        ]
      >

      iex> Scholar.Preprocessing.min_max_scale(Nx.tensor([[1, -1, 2], [3, 0, 0], [0, 1, -1], [2, 3, 1]]), axes: [0], min: 1, max: 3)
      #Nx.Tensor<
        f32[4][3]
        [
          [1.6666667461395264, 1.0, 3.0],
          [3.0, 1.5, 1.6666667461395264],
          [1.0, 2.0, 1.0],
          [2.3333334922790527, 3.0, 2.3333334922790527]
        ]
      >

      iex> Scholar.Preprocessing.min_max_scale(42)
      #Nx.Tensor<
        f32
        0.0
      >
  """
  deftransform min_max_scale(tensor, opts \\ []) do
    min_max_scale_n(tensor, NimbleOptions.validate!(opts, @min_max_schema))
  end

  defnp min_max_scale_n(tensor, opts) do
    if opts[:max] <= opts[:min] do
      raise ArgumentError,
            "expected :max to be greater than :min"
    else
      reduced_max = Nx.reduce_max(tensor, axes: opts[:axes], keep_axes: true)
      reduced_min = Nx.reduce_min(tensor, axes: opts[:axes], keep_axes: true)
      denominator = reduced_max - reduced_min
      denominator = Nx.select(denominator == 0, 1, denominator)
      x_std = (tensor - reduced_min) / denominator
      x_std * (opts[:max] - opts[:min]) + opts[:min]
    end
  end

  @doc """
  Converts a tensor into binary values based on the given threshold.

  ## Options

  #{NimbleOptions.docs(@binarize_schema)}

  ## Examples

      iex> Scholar.Preprocessing.binarize(Nx.tensor([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]))
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.0, 1.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0]
        ]
      >

      iex> Scholar.Preprocessing.binarize(Nx.tensor([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]), threshold: 1.3, type: {:u, 8})
      #Nx.Tensor<
        u8[3][3]
        [
          [0, 0, 1],
          [1, 0, 0],
          [0, 0, 0]
        ]
      >
  """
  deftransform binarize(tensor, opts \\ []) do
    binarize_n(tensor, NimbleOptions.validate!(opts, @binarize_schema))
  end

  defnp binarize_n(tensor, opts) do
    (tensor > opts[:threshold]) |> Nx.as_type(opts[:type])
  end

  @doc """
  Encodes a tensor's values into integers from range 0 to `:num_classes` - 1.

  ## Options

  #{NimbleOptions.docs(@encode_schema)}

  ## Examples

      iex> Scholar.Preprocessing.ordinal_encode(Nx.tensor([3, 2, 4, 56, 2, 4, 2]), num_classes: 4)
      #Nx.Tensor<
        s64[7]
        [1, 0, 2, 3, 0, 2, 0]
      >
  """
  deftransform ordinal_encode(tensor, opts \\ []) do
    ordinal_encode_n(tensor, NimbleOptions.validate!(opts, @encode_schema))
  end

  defnp ordinal_encode_n(tensor, opts) do
    sorted = Nx.sort(tensor)
    num_classes = opts[:num_classes]

    # A mask with a single 1 in every group of equal values
    representative_mask =
      Nx.concatenate([
        sorted[0..-2//1] != sorted[1..-1//1],
        Nx.tensor([1])
      ])

    representative_indices =
      representative_mask
      |> Nx.argsort(direction: :desc)
      |> Nx.slice_along_axis(0, num_classes)

    representative_values = Nx.take(sorted, representative_indices)

    (Nx.new_axis(tensor, 1) ==
       Nx.new_axis(representative_values, 0))
    |> Nx.argmax(axis: 1)
  end

  @doc """
  Encode labels as a one-hot numeric tensor.

  Labels must be integers from 0 to `:num_classes - 1`. If the data does
  not meet the condition, please use `ordinal_encoding/2` first.

  ## Options

  #{NimbleOptions.docs(@encode_schema)}

  ## Examples

      iex> Scholar.Preprocessing.one_hot_encode(Nx.tensor([2, 0, 3, 2, 1, 1, 0]), num_classes: 4)
      #Nx.Tensor<
        u8[7][4]
        [
          [0, 0, 1, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 1, 0],
          [0, 1, 0, 0],
          [0, 1, 0, 0],
          [1, 0, 0, 0]
        ]
      >
  """
  deftransform one_hot_encode(tensor, opts \\ []) do
    one_hot_encode_n(tensor, NimbleOptions.validate!(opts, @encode_schema))
  end

  defnp one_hot_encode_n(tensor, opts) do
    {len} = Nx.shape(tensor)

    if opts[:num_classes] > len do
      raise ArgumentError,
            "expected :num_classes to be at most as length of label vector"
    end

    Nx.new_axis(tensor, -1) == Nx.iota({1, opts[:num_classes]})
  end

  @doc """
  Normalize samples individually to unit norm.

  The zero-tensors cannot be normalized and they stay the same
  after normalization.

  ## Options

  #{NimbleOptions.docs(@normalize_schema)}

  ## Examples

      iex> Scholar.Preprocessing.normalize(Nx.tensor([[0, 0, 0], [3, 4, 5], [-2, 4, 3]]), axes: [1])
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, 0.0, 0.0],
          [0.4242640733718872, 0.5656854510307312, 0.7071067690849304],
          [-0.3713906705379486, 0.7427813410758972, 0.5570860505104065]
        ]
      >

      iex> Scholar.Preprocessing.normalize(Nx.tensor([[0, 0, 0], [3, 4, 5], [-2, 4, 3]]))
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, 0.0, 0.0],
          [0.3375263810157776, 0.4500351846218109, 0.5625439882278442],
          [-0.22501759231090546, 0.4500351846218109, 0.3375263810157776]
        ]
      >
  """
  deftransform normalize(tensor, opts \\ []) do
    normalize_n(tensor, NimbleOptions.validate!(opts, @normalize_schema))
  end

  defnp normalize_n(tensor, opts) do
    shape = Nx.shape(tensor)
    type = to_float_type(tensor)
    zeros = Nx.broadcast(Nx.tensor(0.0, type: type), shape)

    norm =
      case opts[:norm] do
        :euclidean ->
          Scholar.Metrics.Distance.euclidean(tensor, zeros, axes: opts[:axes])

        :manhattan ->
          Scholar.Metrics.Distance.manhattan(tensor, zeros, axes: opts[:axes])

        :chebyshev ->
          Scholar.Metrics.Distance.chebyshev(tensor, zeros, axes: opts[:axes])

        other ->
          raise ArgumentError,
                "expected :norm to be one of: :euclidean, :manhattan, and :chebyshev, got: #{inspect(other)}"
      end

    shape_to_broadcast = unsqueezed_reduced_shape(shape, opts[:axes])

    norm =
      Nx.select(norm == 0.0, Nx.tensor(1.0, type: type), norm) |> Nx.reshape(shape_to_broadcast)

    tensor / norm
  end

  deftransformp unsqueezed_reduced_shape(shape, axes) do
    if axes != nil do
      Enum.reduce(axes, shape, &put_elem(&2, &1, 1))
    else
      Tuple.duplicate(1, Nx.rank(shape))
    end
  end
end
