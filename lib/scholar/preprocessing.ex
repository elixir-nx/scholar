defmodule Scholar.Preprocessing do
  @moduledoc """
  Set of functions for preprocessing data.
  """

  import Nx.Defn

  @doc """
  Standardizes the tensor by removing the mean and scaling to unit variance.

  Formula: `z = (x - u) / s`

  Where `u` is the mean of the samples, and `s` is the standard deviation.
  Standardization can be helpful in cases where the data follows a Gaussian distribution
  (or Normal distribution) without outliers.

  ## Options

    * `:axes` - Axes to standarize a tensor over. By default the
    whole tensor is standarized.

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
  @spec standard_scale(tensor :: Nx.Tensor.t(), opts :: keyword()) :: Nx.Tensor.t()
  defn standard_scale(tensor, opts \\ []) do
    opts = keyword!(opts, [:axes])
    std = Nx.standard_deviation(tensor, axes: opts[:axes], keep_axes: true)
    mean_reduced = Nx.mean(tensor, axes: opts[:axes], keep_axes: true)
    mean_reduced = Nx.select(std == 0, 0.0, mean_reduced)
    (tensor - mean_reduced) / Nx.select(std == 0, 1.0, std)
  end

  @doc """
  Scales a tensor by dividing each sample in batch by maximum absolute value in the batch

  ## Options

    * `:axes` - Axes to scale a tensor over. By default the
    whole tensor is scaled.

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

  @spec max_abs_scale(tensor :: Nx.Tensor.t(), opts :: keyword()) :: Nx.Tensor.t()
  defn max_abs_scale(tensor, opts \\ []) do
    opts = keyword!(opts, [:axes])
    max_abs = Nx.abs(tensor) |> Nx.reduce_max(axes: opts[:axes], keep_axes: true)
    tensor / Nx.select(max_abs == 0, 1, max_abs)
  end

  @doc """
  Transform a tensor by scaling each batch to the given range.

  ## Options

    * `:axes` - Axes to scale a tensor over. By default the
    whole tensor is scaled.

    * `:min` - The lower boundary of the desired range of transformed data.
    Defaults to 0.

    * `:max` - The upper boundary of the desired range of transformed data.
    Defautls to 1.

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

  @spec min_max_scale(tensor :: Nx.Tensor.t(), opts :: keyword()) :: Nx.Tensor.t()
  defn min_max_scale(tensor, opts \\ []) do
    opts = keyword!(opts, [:axes, min: 0, max: 1])

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

    * `:threshold` - Feature values below or equal to this are replaced by 0, above it by 1. Defaults to `0`.

    * `:type` - Type of the resultant tensor. Defaults to `{:f, 32}`.

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

  @spec binarize(tensor :: Nx.Tensor.t(), opts :: Keyword.t()) :: Nx.Tensor.t()
  defn binarize(tensor, opts \\ []) do
    opts = keyword!(opts, threshold: 0, type: {:f, 32})
    (tensor > opts[:threshold]) |> Nx.as_type(opts[:type])
  end

  @doc """
  Encodes a tensor's values into integers from range 0 to `:num_classes - 1`.

  ## Options

    * `:num_classes` - Number of classes to be encoded. Required.

  ## Examples

      iex> Scholar.Preprocessing.ordinal_encode(Nx.tensor([3, 2, 4, 56, 2, 4, 2]), num_classes: 4)
      #Nx.Tensor<
        s64[7]
        [1, 0, 2, 3, 0, 2, 0]
      >
  """

  @spec ordinal_encode(tensor :: Nx.Tensor.t(), opts :: Keyword.t()) :: Nx.Tensor.t()
  defn ordinal_encode(tensor, opts \\ []) do
    {num_samples} = Nx.shape(tensor)
    opts = keyword!(opts, [:num_classes])
    sorted = Nx.sort(tensor)
    num_classes = opts[:num_classes]

    # A mask with a single 1 in every group of equal values
    representative_mask =
      Nx.concatenate([
        Nx.not_equal(sorted[0..-2//1], sorted[1..-1//1]),
        Nx.tensor([1])
      ])

    representative_indices =
      representative_mask
      |> Nx.argsort(direction: :desc)
      |> Nx.slice_along_axis(0, num_classes)

    representative_values = Nx.take(sorted, representative_indices)

    Nx.equal(
      Nx.reshape(tensor, {num_samples, 1}),
      Nx.reshape(representative_values, {1, num_classes})
    )
    |> Nx.argmax(axis: 1)
  end

  @doc """
  Encode labels as a one-hot numeric tensor.

  Labels must be integers from 0 to `:num_classes - 1`. If the data does
  not meet the condition, please use `ordinal_encoding/2` first.

  ## Options

    * `:num_classes` - Number of classes to be encoded. Required.

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

  @spec one_hot_encode(tensor :: Nx.Tensor.t(), opts :: Keyword.t()) :: Nx.Tensor.t()
  defn one_hot_encode(tensor, opts \\ []) do
    Nx.equal(Nx.new_axis(tensor, -1), Nx.iota({1, opts[:num_classes]}))
  end

  @doc """
  Normalize samples individually to unit norm.

  The zero-tensors cannot be normalized and they stay the same
  after normalization.

  ## Options

    * `:axes` - Axes to normalize a tensor over. By default the
    whole tensor is normalized.

    * `:norm` - The norm to use to normalize each non zero sample.
    Possible options are `:euclidean`, `:manhattan`, and `:chebyshev`
    Defaults to `:euclidean`.

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

  @spec normalize(tensor :: Nx.Tensor.t(), opts :: Keyword.t()) :: Nx.Tensor.t()
  defn normalize(tensor, opts \\ []) do
    opts = keyword!(opts, [:axes, norm: :euclidean])
    zeros = Nx.broadcast(0.0, Nx.shape(tensor))

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

    shape = Nx.shape(tensor)
    shape_to_broadcast = unsqueezed_reduced_shape(shape, opts[:axes])
    norm = Nx.select(norm == 0.0, 1.0, norm) |> Nx.reshape(shape_to_broadcast)
    tensor / norm
  end

  deftransformp unsqueezed_reduced_shape(shape, axes) do
    if axes != nil do
      Enum.reduce(axes, shape, &put_elem(&2, &1, 1))
    else
      Tuple.duplicate(1, length(Tuple.to_list(shape)))
    end
  end
end
