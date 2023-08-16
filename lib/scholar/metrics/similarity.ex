defmodule Scholar.Metrics.Similarity do
  @moduledoc """
  Similarity metrics between multi-dimensional tensors.
  """

  import Nx.Defn
  import Scholar.Shared

  general_schema = [
    axis: [
      type: {:custom, Scholar.Options, :axis, []},
      doc: """
      Axis over which the distance will be calculated. By default the distance
      is calculated between the whole tensors.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(general_schema)

  @doc """
  Calculates Jaccard similarity (also known as Jaccard similarity coefficient, or Jaccard index).

  Jaccard similarity is a statistic used to measure similarities between two sets. Mathematically, the calculation
  of Jaccard similarity is the ratio of set intersection over set union.

  #{~S'''
  $$
  J(A, B) = \frac{\mid A \cap B \mid}{\mid A \cup B \mid}
  $$
  '''}
  where A and B are sets.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Examples

      iex> x = Nx.tensor([1.0, 5.0, 3.0, 6.7])
      iex> y = Nx.tensor([5.0, 2.5, 3.1, 9.0])
      iex> Scholar.Metrics.Similarity.jaccard(x, y)
      #Nx.Tensor<
        f32
        0.1428571492433548
      >

      iex> x = Nx.tensor([1, 2, 3, 5, 7])
      iex> y = Nx.tensor([1, 2, 4, 8, 9])
      iex> Scholar.Metrics.Similarity.jaccard(x, y)
      #Nx.Tensor<
        f32
        0.25
      >

      iex> x = Nx.tensor([[0, 1, 2], [3, 4, 5]])
      iex> y = Nx.tensor([[0, 3, 4], [3, 4, 8]])
      iex> Scholar.Metrics.Similarity.jaccard(x, y, axis: 1)
      #Nx.Tensor<
        f32[2]
        [0.20000000298023224, 0.5]
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Similarity.jaccard(x, y)
      #Nx.Tensor<
        f32
        0.6666666865348816
      >
  """
  deftransform jaccard(x, y, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    axis = Keyword.get(opts, :axis)
    axes = if axis == nil, do: axis, else: [axis]
    opts = Keyword.put(opts, :axes, axes)
    jaccard_n(x, y, opts)
  end

  defnp jaccard_n(x, y, opts) do
    {x_size, y_size} =
      case opts[:axis] do
        nil ->
          {unique_size(Nx.flatten(x), opts), unique_size(Nx.flatten(y), opts)}

        _ ->
          {unique_size(x, opts), unique_size(y, opts)}
      end

    result_type = Nx.Type.to_floating(Nx.Type.merge(Nx.type(x), Nx.type(y)))

    union_size =
      case opts[:axis] do
        nil ->
          unique_size(Nx.concatenate([Nx.flatten(x), Nx.flatten(y)]), opts)

        _ ->
          unique_size(Nx.concatenate([x, y], axis: opts[:axis]), opts)
      end

    intersection_size = x_size + y_size - union_size

    (intersection_size / union_size) |> Nx.as_type(result_type)
  end

  @doc """
  Calculates Jaccard similarity based on binary attributes.
  It assumes that inputs have the same shape.

  #{~S'''
  $$
  J(X, Y) = \frac{M\_{11}}{M\_{01} + M\_{10} + M\_{11}}
  $$
  '''}

  Where:

  * $M_{11}$ is the total number of attributes, for which both $X$ and $Y$ have 1.
  * $M_{10}$ is the total number of attributes, for which $X$ has 1 and $Y$ has 0.
  * $M_{01}$ is the total number of attributes, for which $X$ has 0 and $Y$ has 1.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Examples

      iex> x = Nx.tensor([1,0,0,1,1,1])
      iex> y = Nx.tensor([0,0,1,1,1,0])
      iex> Scholar.Metrics.Similarity.binary_jaccard(x, y)
      #Nx.Tensor<
        f32
        0.4000000059604645
      >

      iex> x = Nx.tensor([[1,1,0,1], [1,1,0,1]])
      iex> y = Nx.tensor([[1,1,0,1], [1,0,0,1]])
      iex> Scholar.Metrics.Similarity.binary_jaccard(x, y)
      #Nx.Tensor<
        f32
        0.8333333134651184
      >

      iex> x = Nx.tensor([[1,1,0,1], [1,1,0,1]])
      iex> y = Nx.tensor([[1,1,0,1], [1,0,0,1]])
      iex> Scholar.Metrics.Similarity.binary_jaccard(x, y, axis: 1)
      #Nx.Tensor<
        f32[2]
        [1.0, 0.6666666865348816]
      >

      iex> x = Nx.tensor([1, 1])
      iex> y = Nx.tensor([1, 0, 0])
      iex> Scholar.Metrics.Similarity.binary_jaccard(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}
  """
  deftransform binary_jaccard(x, y, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    axis = Keyword.get(opts, :axis)
    axis = if axis == nil, do: axis, else: [axis]
    opts = Keyword.put(opts, :axis, axis)
    binary_jaccard_n(x, y, opts)
  end

  defnp binary_jaccard_n(x, y, opts) do
    # We're requiring the same shape because usual use cases will have the same shape.
    # The last axis could in theory be different on both sides.
    assert_same_shape!(x, y)

    m11 = Nx.sum(x and y, axes: opts[:axis])
    m10 = Nx.sum(x > y, axes: opts[:axis])
    m01 = Nx.sum(x < y, axes: opts[:axis])

    result_type = Nx.Type.to_floating(Nx.Type.merge(Nx.type(x), Nx.type(y)))

    (m11 / (m11 + m10 + m01)) |> Nx.as_type(result_type)
  end

  @doc """
  Calculates Dice coefficient.

  It is a statistic used to gauge the similarity of two samples.
  Mathematically, it is defined as:

  #{~S'''
  $$
  Dice(A, B) = \frac{2 \mid A \cap B \mid}{\mid A \mid + \mid B \mid}
  $$
  '''}
  where A and B are sets.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Examples

      iex> x = Nx.tensor([1.0, 5.0, 3.0, 6.7])
      iex> y = Nx.tensor([5.0, 2.5, 3.1, 9.0])
      iex> Scholar.Metrics.Similarity.dice_coefficient(x, y)
      #Nx.Tensor<
        f32
        0.25
      >

      iex> x = Nx.tensor([1, 2, 3, 5, 7])
      iex> y = Nx.tensor([1, 2, 4, 8, 9])
      iex> Scholar.Metrics.Similarity.dice_coefficient(x, y)
      #Nx.Tensor<
        f32
        0.4000000059604645
      >

      iex> x = Nx.iota({2,3})
      iex> y = Nx.tensor([[0, 3, 4], [3, 4, 8]])
      iex> Scholar.Metrics.Similarity.dice_coefficient(x, y, axis: 1)
      #Nx.Tensor<
        f32[2]
        [0.3333333134651184, 0.6666666865348816]
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Similarity.dice_coefficient(x, y)
      #Nx.Tensor<
        f32
        0.800000011920929
      >
  """
  defn dice_coefficient(x, y, opts \\ []) do
    j = jaccard(x, y, opts)
    2 * j / (1 + j)
  end

  @doc """
  Calculates Dice coefficient based on binary attributes.
  It assumes that inputs have the same shape.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Examples

      iex> x = Nx.tensor([1,0,0,1,1,1])
      iex> y = Nx.tensor([0,0,1,1,1,0])
      iex> Scholar.Metrics.Similarity.dice_coefficient_binary(x, y)
      #Nx.Tensor<
        f32
        0.5714285969734192
      >

      iex> x = Nx.tensor([[1,1,0,1], [1,1,0,1]])
      iex> y = Nx.tensor([[1,1,0,1], [1,0,0,1]])
      iex> Scholar.Metrics.Similarity.dice_coefficient_binary(x, y)
      #Nx.Tensor<
        f32
        0.9090909361839294
      >

      iex> x = Nx.tensor([[1,1,0,1], [1,1,0,1]])
      iex> y = Nx.tensor([[1,1,0,1], [1,0,0,1]])
      iex> Scholar.Metrics.Similarity.dice_coefficient_binary(x, y, axis: 1)
      #Nx.Tensor<
        f32[2]
        [1.0, 0.800000011920929]
      >

      iex> x = Nx.tensor([1, 1])
      iex> y = Nx.tensor([1, 0, 0])
      iex> Scholar.Metrics.Similarity.dice_coefficient_binary(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}
  """
  defn dice_coefficient_binary(x, y, opts \\ []) do
    j = binary_jaccard(x, y, opts)
    2 * j / (1 + j)
  end

  defnp unique_size(%Nx.Tensor{shape: shape} = tensor, opts) do
    case shape do
      {} ->
        raise "expected tensor to have at least one dimension, got scalar"

      {1} ->
        1

      _ ->
        different_from_successor? =
          case opts[:axis] do
            nil ->
              tensor = Nx.sort(tensor)
              Nx.diff(tensor) != 0

            axis ->
              tensor = Nx.sort(tensor, axis: opts[:axis])
              Nx.diff(tensor, axis: axis) != 0
          end

        Nx.sum(different_from_successor?, axes: opts[:axes]) + 1
    end
  end
end
