defmodule Scholar.Covariance do
  @moduledoc """
  Algorithms to estimate the covariance of features given a set of points.
  """
  import Nx.Defn

  opts = [
    assume_centered: [
      type: :boolean,
      default: false,
      doc: """
      If `true`, data will not be centered before computation.
      Useful when working with data whose mean is almost, but not exactly zero.
      If `false`, data will be centered before computation.
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

  @doc """
  Computes covariance matrix for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Example
  iex> Scholar.Covariance.covariance_matrix(Nx.tensor([[3,6,5], [26,75,3], [23,4,1]]))
  #Nx.Tensor<
    f32[3][3]
    [
      [104.22222137451172, 195.5555419921875, -13.333333015441895],
      [195.5555419921875, 1089.5555419921875, 1.3333333730697632],
      [-13.333333015441895, 1.3333333730697632, 2.6666667461395264]
    ]
  >

  """
  deftransform covariance_matrix(x, opts \\ []) do
    covariance_matrix_n(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp covariance_matrix_n(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError, "expected data to have rank equal 2, got: #{inspect(Nx.rank(x))}"
    end

    {num_samples, _num_features} = Nx.shape(x)
    x = if opts[:assume_centered], do: x, else: x - Nx.mean(x, axes: [0])
    matrix = Nx.dot(Nx.transpose(x), x)

    if opts[:biased] do
      matrix / num_samples
    else
      matrix / (num_samples - 1)
    end
  end
end
