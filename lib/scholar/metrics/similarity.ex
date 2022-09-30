defmodule Scholar.Metrics.Similarity do
  @moduledoc """
  Similarity metrics between 1-D tensors.
  """

  import Nx.Defn
  import Scholar.Shared

  @doc ~S"""
  Calculates Jaccard similarity (also known as Jaccard similarity coefficient, or Jaccard index).

  Jaccard similarity is a statistic used to measure similarities between two sets. Mathematically, the calculation
  of Jaccard similarity is the ratio of set intersection over set union.

  $$
  J(A, B) = \frac{\mid A \cap B \mid}{\mid A \cup B \mid}
  $$

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

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Similarity.jaccard(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}
  """
  defn jaccard(x, y) do
    # We're requiring the same shape because usual use cases will have the same shape.
    # The last axis could in theory be different on both sides.
    assert_same_shape!(x, y)

    x_size = unique_size(x)
    y_size = unique_size(y)

    union_size = unique_size(Nx.concatenate([x, y]))
    intersection_size = x_size + y_size - union_size

    intersection_size / union_size
  end

  @doc ~S"""
  Calculates Jaccard similarity based on binary attributes.

  $$
  J = \frac{M\_{11}}{M\_{01} + M\_{10} + M\_{11}}
  $$

  $M\_{11}$ is the total numbers of attributes, for which both X and Y have 1.\
  $M\_{10}$ is the total numbers of attributes, for which X has 1 and Y has 0.\
  $M\_{01}$ is the total numbers of attributes, for which X has 0 and Y has 1.\
  $M\_{00}$ is the total numbers of attributes, for which both X and Y have 0.

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

      iex> x = Nx.tensor([1, 1])
      iex> y = Nx.tensor([1, 0, 0])
      iex> Scholar.Metrics.Similarity.binary_jaccard(x, y)
      ** (ArgumentError) expected tensor to have shape {2}, got tensor with shape {3}
  """
  defn binary_jaccard(x, y) do
    # We're requiring the same shape because usual use cases will have the same shape.
    # The last axis could in theory be different on both sides.
    assert_same_shape!(x, y)

    m11 = Nx.sum(x and y)
    m10 = Nx.sum(x > y)
    m01 = Nx.sum(x < y)

    m11 / (m11 + m10 + m01)
  end

  defnp unique_size(%Nx.Tensor{shape: shape} = tensor) do
    case shape do
      {} ->
        raise "expected input shape of at least {1}, got: {}"

      {1} ->
        1

      _ ->
        sorted = Nx.sort(tensor)

        different_from_successor? = sorted[0..-2//1] != sorted[1..-1//1]

        Nx.sum(different_from_successor?) + 1
    end
  end
end
