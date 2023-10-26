defmodule Scholar.Linear.PolynomialRegression do
  @moduledoc """
  Least squares polynomial regression.

  Time complexity of polynomial regression is $O((K^2) * (K+N))$ where $N$ is the number of samples and $K$ is the number of features.
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :intercept, :degree]}
  defstruct [:coefficients, :intercept, :degree]

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
      The degree of the feature matrix to return. Must be an integer equal or greater than 1. 1
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

  transform_opts = Keyword.take(opts, [:degree, :fit_intercept?])

  @opts_schema NimbleOptions.new!(opts)
  @transform_opts_schema NimbleOptions.new!(transform_opts)

  @doc """
  Fits a polynomial regression model for sample inputs `x` and
  sample targets `y`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:coefficients` - Estimated coefficients for the polynomial regression problem.

    * `:intercept` - Independent term in the polynomial model.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> model = Scholar.Linear.PolynomialRegression.fit(x, y, degree: 1)
      iex> model.coefficients
      #Nx.Tensor<
        f32[2]
        [-0.49724727869033813, -0.7010392546653748]
      >
      iex> model.intercept
      #Nx.Tensor<
        f32
        5.896470069885254
      >
      iex> model.degree
      1

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> model = Scholar.Linear.PolynomialRegression.fit(x, y, degree: 2)
      iex> model.coefficients
      #Nx.Tensor<
        f32[5]
        [-0.021396497264504433, -0.004854594357311726, -0.0884987860918045, -0.062211357057094574, -0.04369127005338669]
      >
      iex> model.intercept
      #Nx.Tensor<
        f32
        4.418517112731934
      >
      iex> model.degree
      2
  """
  deftransform fit(x, y, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    x_transform = transform(x, fit_intercept?: false, degree: opts[:degree])

    linear_reg =
      Scholar.Linear.LinearRegression.fit(x_transform, y, Keyword.take(opts, [:fit_intercept?]))

    %__MODULE__{
      coefficients: linear_reg.coefficients,
      intercept: linear_reg.intercept,
      degree: opts[:degree]
    }
  end

  @doc """
  Makes predictions with the given `model` on input `x`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> model = Scholar.Linear.PolynomialRegression.fit(x, y, degree: 2)
      iex> Scholar.Linear.PolynomialRegression.predict(model, Nx.tensor([[2.0, 1.0]]))
      #Nx.Tensor<
        f32[1]
        [3.8487603664398193]
      >
  """
  deftransform predict(model, x) do
    Scholar.Linear.LinearRegression.predict(
      %Scholar.Linear.LinearRegression{
        coefficients: model.coefficients |> Nx.flatten(),
        intercept: model.intercept
      },
      transform_n(x, degree: model.degree, fit_intercept?: false)
    )
  end

  @doc """
  Computes the feature matrix for polynomial regression.

  #{NimbleOptions.docs(@transform_opts_schema)}

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
    transform_n(x, NimbleOptions.validate!(opts, @transform_opts_schema))
  end

  deftransformp transform_n(x, opts) do
    {_n_samples, n_features} = Nx.shape(x)

    x_split = Enum.map(0..(n_features - 1), &get_column(x, &1))

    Enum.scan(
      0..(opts[:degree] - 1)//1,
      nil,
      fn
        0, nil ->
          x_split

        _, prev_degree ->
          compute_degree(x, prev_degree)
      end
    )
    |> List.flatten()
    |> Nx.concatenate(axis: 1)
    |> add_intercept(opts)
  end

  deftransformp compute_degree(x, previous_degree) do
    {_n_samples, n_features} = Nx.shape(x)

    Enum.map(0..(n_features - 1), fn nf ->
      previous_degree
      |> Enum.slice(nf..-1//1)
      |> Nx.concatenate(axis: 1)
      |> compute_column(x, nf)
    end)
  end

  defnp get_column(x, n) do
    Nx.slice_along_axis(x, n, 1, axis: 1)
  end

  defnp compute_column(previous, x, n) do
    get_column(x, n) * previous
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
