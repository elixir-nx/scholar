defmodule Scholar.Linear.PolynomialRegression do
  @moduledoc """
  Least squares polynomial regression.
  """

  import Nx.Defn

  # @derive {Nx.Container, containers: [:coefficients, :intercept]}
  # defstruct [:coefficients, :intercept]

  opts = [
    sample_weights: [
      type: {:list, {:custom, Scholar.Options, :positive_number, []}},
      doc: """
      The weights for each observation. If not provided,
      all observations are assigned equal weight.
      """
    ],
    degree: [
      type: :pos_integer,
      default: 2,
      doc: """
      The degree of the feature matrix to return. Must be a >1 integer. 1
      returns the input matrix.
      """
    ],
    fit_intercept?: [
      type: :boolean,
      default: true,
      doc: """
      If set to `true`, a model will fit the intercept. Otherwise,
      the intercept is set to `0.0`. The intercept is an independent term
      in a linear model. Specifically, it is the expected mean value
      of targets for a zero-vector on input.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a polynomial regression model for sample inputs `a` and
  sample targets `b`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:coefficients` - Estimated coefficients for the polynomial regression problem.

    * `:intercept` - Independent term in the linear model.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> Scholar.Linear.PolynomialRegression.fit(x, y, degree: 1)
      %Scholar.Linear.LinearRegression{
        coefficients: #Nx.Tensor<
          f32[2]
          [-0.4972473084926605, -0.70103919506073]
        >, intercept: #Nx.Tensor<
          f32
          5.896470069885254
        >
      }

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> model = Scholar.Linear.PolynomialRegression.fit(x, y, degree: 2)
      %Scholar.Linear.LinearRegression{
        coefficients: #Nx.Tensor<
          f32[5]
          [-0.021396497264504433, -0.004854593891650438, -0.08849877119064331, -0.062211357057094574, -0.04369127377867699]
        >, intercept: #Nx.Tensor<
          f32
          4.418517112731934
        >
      }
  """
  deftransform fit(a, b, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    a_transform = transform(a, opts |> Keyword.put(:fit_intercept?, false))
    Scholar.Linear.LinearRegression.fit(a_transform, b, Keyword.delete(opts, :degree))
  end

  @doc """
  Computes the feature matrix for polynomial regression.

  #{NimbleOptions.docs(@opts_schema)}

  ## Examples

      iex> x = Nx.tensor([[2]])
      iex> Scholar.Linear.PolynomialRegression.transform(x, degree: 0)
      ** (NimbleOptions.ValidationError) invalid value for :degree option: expected positive integer, got: 0

      iex> x = Nx.tensor([[2]])
      iex> Scholar.Linear.PolynomialRegression.transform(x, degree: 5, fit_intercept?: false)
      #Nx.Tensor<
        s64[1][5]
        [
          [2, 4, 8, 16, 32]
        ]
      >

      iex> x = Nx.tensor([[2, 3]])
      iex> Scholar.Linear.PolynomialRegression.transform(x)
      #Nx.Tensor<
        s64[1][6]
        [
          [1, 2, 3, 4, 6, 9]
        ]
      >

      iex> x = Nx.iota({3, 2})
      iex> Scholar.Linear.PolynomialRegression.transform(x, fit_intercept?: false)
      #Nx.Tensor<
        s64[3][5]
        [
          [0, 1, 0, 0, 1],
          [2, 3, 4, 6, 9],
          [4, 5, 16, 20, 25]
        ]
      >
  """
  deftransform transform(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    transform_n(x, opts)
  end

  deftransform transform_n(x, opts) do
    x_split = initial_data_split(x)

    1..(opts[:degree] - 1)//1
    |> Enum.reduce([x_split], fn _, prev_degree ->
      [prev_degree, compute_degree(x, List.last(prev_degree))]
    end)
    |> List.flatten()
    |> Nx.concatenate(axis: 1)
    |> add_intercept(opts)
  end

  @spec initial_data_split(Nx.Tensor) :: list(Nx.Tensor)
  deftransform initial_data_split(x) do
    {n_samples, n_features} = Nx.shape(x)

    Enum.map(
      0..(n_features - 1),
      fn n ->
        Nx.reshape(x[[0..-1//1, n]], {n_samples, :auto})
      end
    )
  end

  @spec compute_degree(Nx.Tensor, list(Nx.Tensor)) :: list(Nx.Tensor)
  deftransform compute_degree(x, previous_degree) do
    {_n_samples, n_features} = Nx.shape(x)

    Enum.map(0..(n_features - 1), fn nf ->
      previous_degree
      |> Enum.slice(nf..-1)
      |> Nx.concatenate(axis: 1)
      |> compute_column(x, nf)
    end)
  end

  defnp compute_column(previous, x, n) do
    {n_samples, _n_features} = Nx.shape(x)

    x[[0..-1//1, n]]
    |> Nx.reshape({n_samples, :auto})
    |> Nx.multiply(previous)
  end

  defnp add_intercept(x, opts) do
    if opts[:fit_intercept?] do
      {n_samples, _n_features} = Nx.shape(x)

      [Nx.broadcast(1, {n_samples, 1}), x]
      |> Nx.concatenate(axis: 1)
    else
      x
    end
  end
end
