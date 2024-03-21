defmodule Scholar.Linear.BayesianRidgeRegression do
  require Nx
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:coefficients, :intercept]}
  defstruct [:coefficients, :intercept]
  opts = [
    sample_weights: [
      type:
        {:or,
         [
           {:custom, Scholar.Options, :non_negative_number, []},
           {:list, {:custom, Scholar.Options, :non_negative_number, []}},
           {:custom, Scholar.Options, :weights, []}
         ]},
      doc: """
      The weights for each observation. If not provided,
      all observations are assigned equal weight.
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
    ],
    solver: [
      type: {:in, [:svd, :cholesky]},
      default: :svd,
      doc: """
      Solver to use in the computational routines:

      * `:svd` - Uses a Singular Value Decomposition of A to compute the Ridge coefficients.
      In particular, it is more stable for singular matrices than `:cholesky` at the cost of being slower.

      * `:cholesky` - Uses the standard `Nx.LinAlg.solve` function to obtain a closed-form solution.
      """
    ],
    alpha: [
      type:
        {:or,
         [
           {:custom, Scholar.Options, :non_negative_number, []},
           {:list, {:custom, Scholar.Options, :non_negative_number, []}},
           {:custom, Scholar.Options, :weights, []}
         ]},
      default: 1.0,
      doc: ~S"""
      Constant that multiplies the $L_2$ term, controlling regularization strength.
      `:alpha` must be a non-negative float i.e. in [0, inf).

      If `:alpha` is set to 0.0 the objective is the ordinary least squares regression.
      In this case, for numerical reasons, you should use `Scholar.Linear.LinearRegression` instead.
      """
    ]
  ]
  @opts_schema NimbleOptions.new!(opts)
  deftransform fit(x, y, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)    
        {sample_weights, opts} = Keyword.pop(opts, :sample_weights, 1.0)
    x_type = to_float_type(x)

    sample_weights =
      if Nx.is_tensor(sample_weights),
        do: Nx.as_type(sample_weights, x_type),
        else: Nx.tensor(sample_weights, type: x_type)

    {alpha, opts} = Keyword.pop!(opts, :alpha)
    alpha = Nx.tensor(alpha, type: x_type) |> Nx.flatten()
    num_targets = if Nx.rank(y) == 1, do: 1, else: Nx.axis_size(y, 1)

    if Nx.size(alpha) not in [0, 1, num_targets] do
      raise ArgumentError,
      "expected number of targets be the same as number of penalties, got: #{inspect(num_targets)} != #{inspect(Nx.size(alpha))}"
    end

    fit_n(x, y, sample_weights, alpha, opts)
  end

  defnp fit_n(a, b, sample_weights, alpha, opts) do
    {u, s, vh} = Nx.LinAlg.svd(a, full_matrices?: false)
    eigen_vals = Nx.pow(s, 2)
    %__MODULE__{coefficients: Nx.tensor([1, 2, 3]), intercept: Nx.tensor([1])}
  end

  defnp update_coef(
          x, y, n_samples, n_features, xt_y,
          u, vh, eigen_vals,
          alpha, lambda) do
    regularization = vh / (eigen_vals + lambda / alpha)
    reg_transpose = Nx.dot(regularization, xt_y)
    coef = Nx.dot(Nx.transpose(vh), reg_transpose)

    error = y - Nx.dot(x, coef)
    squared_error = error ** 2
    rmse = Nx.sum(squared_error)

    {coef, rmse}
  end
end
