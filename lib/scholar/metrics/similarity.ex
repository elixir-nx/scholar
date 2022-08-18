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
      ** (ArgumentError) expected input shapes to be equal, got {2} != {3}
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

  defnp unique_size(%Nx.Tensor{shape: shape} = tensor) do
    case shape do
      {} ->
        raise "expected input shape of at least {1}, got: {}"

      {1} ->
        Nx.tensor(1)

      _ ->
        sorted = Nx.sort(tensor)

        different_from_successor? = Nx.not_equal(sorted[0..-2//1], sorted[1..-1//1])

        different_from_successor?
        |> Nx.sum()
        |> Nx.add(1)
    end
  end
end
