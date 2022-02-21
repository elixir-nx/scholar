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
  """
  @spec euclidean(Nx.t(), Nx.t()) :: Nx.t()
  defn euclidean(x, y) do
    assert_same_shape!(x, y)
    diff = x - y

    cond do
      Nx.all(diff == 0) ->
        Nx.tensor(0.0)

      true ->
        diff
        |> Nx.LinAlg.norm()
        |> Nx.as_type({:f, 32})
    end
  end

  @doc """
  Squared euclidean distance.

  $$
  D(x, y) = \\sum_i (x_i - y_i)^2
  $$

  ## Examples

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([3, 2])
      iex> Scholar.Metrics.Distance.squared_euclidean(x, y)
      #Nx.Tensor<
        f32
        4.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2])
      iex> Scholar.Metrics.Distance.squared_euclidean(x, y)
      #Nx.Tensor<
        f32
        0.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Distance.squared_euclidean(x, y)
      ** (ArgumentError) expected input shapes to be equal, got {2} != {3}
  """
  @spec squared_euclidean(Nx.t(), Nx.t()) :: Nx.t()
  defn squared_euclidean(x, y) do
    assert_same_shape!(x, y)

    x
    |> Nx.subtract(y)
    |> Nx.power(2)
    |> Nx.sum()
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Manhattan, taxicab, or l1 distance.

  $$
  D(x, y) = \\sum_i |x_i - y_i|
  $$

  ## Examples

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([3, 2])
      iex> Scholar.Metrics.Distance.manhattan(x, y)
      #Nx.Tensor<
        f32
        2.0
      >

      iex> x = Nx.tensor([1, 2])
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
  """
  @spec manhattan(Nx.t(), Nx.t()) :: Nx.t()
  defn manhattan(x, y) do
    assert_same_shape!(x, y)

    x
    |> Nx.subtract(y)
    |> Nx.abs()
    |> Nx.sum()
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Chebyshev or l-infinity distance.

  $$
  D(x, y) = \\max_i |x_i - y_i|
  $$

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
  """
  @spec chebyshev(Nx.t(), Nx.t()) :: Nx.t()
  defn chebyshev(x, y) do
    assert_same_shape!(x, y)

    x
    |> Nx.subtract(y)
    |> Nx.LinAlg.norm(ord: :inf)
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Minkowski distance.

  $$
  D(x, y) = \\left(\\sum_i |x_i - y_i|^p\\right)^{\\frac{1}{p}}
  $$

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
  """
  @spec minkowski(Nx.t(), Nx.t(), integer()) :: Nx.t()
  defn minkowski(x, y, p \\ 2) do
    assert_same_shape!(x, y)

    x
    |> Nx.subtract(y)
    |> Nx.abs()
    |> Nx.power(p)
    |> Nx.sum()
    |> Nx.power(1.0 / p)
  end

  @doc """
  Cosine distance.

  $$
  1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}
  $$

  ## Examples

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([5, 2])
      iex> Scholar.Metrics.Distance.cosine(x, y)
      #Nx.Tensor<
        f32
        0.2525906562805176
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2])
      iex> Scholar.Metrics.Distance.cosine(x, y)
      #Nx.Tensor<
        f32
        0.0
      >

      iex> x = Nx.tensor([1, 2])
      iex> y = Nx.tensor([1, 2, 3])
      iex> Scholar.Metrics.Distance.cosine(x, y)
      ** (ArgumentError) expected input shapes to be equal, got {2} != {3}
  """
  @spec cosine(Nx.t(), Nx.t()) :: Nx.t()
  defn cosine(x, y) do
    assert_same_shape!(x, y)
    norm_x = Nx.LinAlg.norm(x)
    norm_y = Nx.LinAlg.norm(y)

    cond do
      norm_x == 0.0 and norm_y == 0.0 ->
        0.0

      norm_x == 0.0 or norm_y == 0.0 ->
        1.0

      true ->
        numerator = Nx.dot(x, y)
        denominator = norm_x * norm_y
        1.0 - numerator / denominator
    end
  end
end
