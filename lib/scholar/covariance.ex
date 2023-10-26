defmodule Scholar.Covariance do
  @moduledoc ~S"""
  Algorithms to estimate the covariance of features given a set of points.

  Time complexity of covariance estimation is $O(N * K^2)$ where $N$ is the number of samples
  and $K$ is the number of features.
  """
  import Nx.Defn

  opts = [
    center: [
      type: :boolean,
      default: true,
      doc: """
      If `true`, data will be centered before computation.
      If `false`, data will not be centered before computation.
      Useful when working with data whose mean is almost, but not exactly zero.
      """
    ],
    biased: [
      type: :boolean,
      default: true,
      doc: """
      If `true`, the matrix will be computed using biased covariation. If `false`,
      algorithm uses unbiased covariation.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @deprecated "Use Nx.convariance/2 instead"
  @doc """
  Computes covariance matrix for sample inputs `x`.

  The value on the position $Cov_{ij}$ in the $Cov$ matrix is calculated using the formula:

  #{~S'''
  $$ Cov(X\_i, X\_j) = \frac{\sum\_{k}\left(x\_k -
  \bar{x}\right)\left(y\_k - \bar{y}\right)}{N - 1}
  $$
  Where:
    * $X_i$ is a $i$th row of input

    * $x_k$ is a $k$th value of $X_i$

    * $y_k$ is a $k$th value of $X_j$

    * $\bar{x}$ is the mean of $X_i$

    * $\bar{y}$ is the mean of $X_j$

    * $N$ is the number of samples

  This is a non-biased version of covariance.
  The biased version has $N$ in denominator instead of $N - 1$.
  '''}

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Example

      iex> Scholar.Covariance.covariance_matrix(Nx.tensor([[3, 6, 5], [26, 75, 3], [23, 4, 1]]))
      #Nx.Tensor<
        f32[3][3]
        [
          [104.22222137451172, 195.5555419921875, -13.333333015441895],
          [195.5555419921875, 1089.5555419921875, 1.3333333730697632],
          [-13.333333015441895, 1.3333333730697632, 2.6666667461395264]
        ]
      >

      iex> Scholar.Covariance.covariance_matrix(Nx.tensor([[3, 6], [2, 3], [7, 9], [5, 3]]))
      #Nx.Tensor<
        f32[2][2]
        [
          [3.6875, 3.1875],
          [3.1875, 6.1875]
        ]
      >

      iex> Scholar.Covariance.covariance_matrix(Nx.tensor([[3, 6, 5], [26, 75, 3], [23, 4, 1]]),
      ...>   biased: false
      ...> )
      #Nx.Tensor<
        f32[3][3]
        [
          [156.3333282470703, 293.33331298828125, -20.0],
          [293.33331298828125, 1634.333251953125, 2.0],
          [-20.0, 2.0, 4.0]
        ]
      >
  """
  deftransform covariance_matrix(x, opts \\ []) do
    covariance_matrix_n(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp covariance_matrix_n(x, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError, "expected data to have rank equal 2, got: #{inspect(Nx.rank(x))}"
    end

    num_samples = Nx.axis_size(x, 0)
    x = if opts[:center], do: x - Nx.mean(x, axes: [0]), else: x
    matrix = Nx.dot(x, [0], x, [0])

    if opts[:biased] do
      matrix / num_samples
    else
      matrix / (num_samples - 1)
    end
  end

  @deprecated "Use Scholar.Stats.correlation_matrix/2 instead"
  @doc """
  Computes correlation matrix for sample inputs `x`.

  The value on the position $Corr_{ij}$ in the $Corr$ matrix is calculated using the formula:
  #{~S'''
  $$ Corr(X\_i, X\_j) = \frac{Cov(X\_i, X\_j)}{\sqrt{Cov(X\_i, X\_i)Cov(X\_j, X\_j)}} $$
  Where:
    * $X_i$ is a $i$th row of input

    * $Cov(X\_i, X\_j)$ is covariance between features $X_i$ and $X_j$
  '''}

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Example

      iex> Scholar.Covariance.correlation_matrix(Nx.tensor([[3, 6, 5], [26, 75, 3], [23, 4, 1]]))
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.580316960811615, -0.7997867465019226],
          [0.580316960811615, 1.0, 0.024736011400818825],
          [-0.7997867465019226, 0.024736011400818825, 1.0]
        ]
      >

      iex> Scholar.Covariance.correlation_matrix(Nx.tensor([[3, 6], [2, 3], [7, 9], [5, 3]]))
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 0.6673083305358887],
          [0.6673083305358887, 1.0]
        ]
      >

      iex> Scholar.Covariance.correlation_matrix(Nx.tensor([[3, 6, 5], [26, 75, 3], [23, 4, 1]]),
      ...>   biased: false
      ...> )
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.5803170204162598, -0.7997867465019226],
          [0.5803170204162598, 1.0, 0.024736013263463974],
          [-0.7997867465019226, 0.024736013263463974, 1.0]
        ]
      >
  """

  deftransform correlation_matrix(x, opts \\ []) do
    correlation_matrix_n(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp correlation_matrix_n(x, opts) do
    variances =
      if opts[:biased] do
        Nx.variance(x, axes: [0])
      else
        Nx.variance(x, axes: [0], ddof: 1)
      end

    Scholar.Covariance.covariance_matrix(x, opts) /
      Nx.sqrt(Nx.new_axis(variances, 1) * Nx.new_axis(variances, 0))
  end
end
