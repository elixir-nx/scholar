defmodule Scholar.Polynomial do
  @moduledoc """
  Set of functions for polynomial regression transformations.
  """

  import Nx.Defn

  transform_schema = [
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

  @transform_schema NimbleOptions.new!(transform_schema)

  @doc """
    Computes the feature matrix for polynomial regression.

    #{NimbleOptions.docs(@transform_schema)}

    ## Examples
      iex> x = Nx.tensor([[2]])
      iex> Polynomial.transform(x, degree: 0)
      ** (NimbleOptions.ValidationError) invalid value for :degree option: expected positive integer, got: 0

      iex> x = Nx.tensor([[2]])
      iex> Polynomial.transform(x, degree: 5, fit_intercept?: false)
      #Nx.Tensor<
        s64[1][5]
        [
          [2, 4, 8, 16, 32]
        ]
      >

      iex> x = Nx.tensor([[2, 3], [1, 3]])
      iex> Polynomial.transform(x, degree: 1, fit_intercept?: false)
      #Nx.Tensor<
        s64[2][2]
        [
          [2, 3],
          [1, 3]
        ]
      >

      iex> x = Nx.tensor([[2, 3]])
      iex> Polynomial.transform(x)
      #Nx.Tensor<
        s64[1][6]
        [
          [1, 2, 3, 4, 6, 9]
        ]
      >

      iex> x = Nx.iota({3, 2})
      iex> Polynomial.transform(x, fit_intercept?: false)
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
    opts = NimbleOptions.validate!(opts, @transform_schema)
    {n_samples, n_features} = Nx.shape(x)

    x_split = Enum.map(0..(n_features - 1), &Nx.reshape(x[[.., &1]], {n_samples, :auto}))

    polynomial_features =
      if opts[:degree] != 1 do
        2..opts[:degree]
        |> Enum.reduce([x_split], fn _, prev_degree -> compute_degree(x, prev_degree) end)
        |> List.flatten()
        |> Nx.concatenate(axis: 1)
      else
        x
      end

    if opts[:fit_intercept?], do: add_intercept(polynomial_features), else: polynomial_features
  end

  defp compute_degree(x, previous_degree) do
    {_n_samples, n_features} = Nx.shape(x)

    res =
      0..(n_features - 1)
      |> Enum.map(fn nf ->
        previous_joined =
          previous_degree
          |> List.last()
          |> Enum.slice(nf..-1)
          |> Nx.concatenate(axis: 1)

        compute_column(x, previous_joined, nf)
      end)

    [previous_degree, res]
  end

  defnp compute_column(x, previous, n) do
    {n_samples, _n_features} = Nx.shape(x)

    Nx.reshape(x[[0..-1//1, n]], {n_samples, :auto})
    |> Nx.multiply(previous)
  end

  defp add_intercept(x) do
    {n_samples, _n_features} = Nx.shape(x)

    [Nx.broadcast(1, {n_samples, 1}) | [x]]
    |> Nx.concatenate(axis: 1)
  end
end
