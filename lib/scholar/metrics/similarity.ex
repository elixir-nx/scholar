defmodule Scholar.Metrics.Similarity do
  @moduledoc """
  Similarity metrics between 1-D tensors.
  """

  import Nx.Defn
  import Scholar.Shared

  @doc """
  The Jaccard similarity (also known as Jaccard similarity coefficient, or Jaccard index)
  is a statistic used to measure similarities between two sets. Mathematically, the calculation
  of Jaccard similarity is the ratio of set intersection over set union.

  J(A, B) = |A∩B| / |A∪B|

  ## Examples

      iex> x = Nx.tensor([1.0, 5.0, 3.0, 6.7])
      iex> y = Nx.tensor([5.0, 2.5, 3.1, 9.0])
      iex> Scholar.Metrics.Similarity.jaccard(x, y)
      #Nx.Tensor<
        f32
        0.1428571492433548
      >

      iex> x = Nx.tensor([1,2,3,5,7])
      iex> y = Nx.tensor([1,2,4,8,9])
      iex> Scholar.Metrics.Similarity.jaccard x, y
      #Nx.Tensor<
        f32
        0.25
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Similarity.jaccard x, y
      ** (ArgumentError) expected input shapes to be equal, got {2} != {3}
  """
  defn jaccard(x, y) do
    assert_same_shape!(x, y)

    x_size = unique_size(x)
    y_size = unique_size(y)

    union_size = unique_size(Nx.concatenate([x, y]))
    intersection_size = Nx.add(x_size, y_size) - union_size

    Nx.divide(intersection_size, union_size)
  end

  defnp unique_size(tensor) do
    sorted = Nx.sort(tensor)

    [
      Nx.not_equal(sorted[0..-2//1], sorted[1..-1//1]),
      Nx.tensor([1])
    ]
    |> Nx.concatenate()
    |> Nx.sum()
  end
end
