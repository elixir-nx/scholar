defmodule Scholar.Linear.BayesianRidgeRegression do
  require Nx
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:coefficients]}
  defstruct [:coefficients]

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
    ],
    alpha_1: [
      type:
        {:or,
         [
           {:custom, Scholar.Options, :non_negative_number, []},
           {:list, {:custom, Scholar.Options, :non_negative_number, []}},
           {:custom, Scholar.Options, :weights, []}
         ]},
      default: 1.0e-6,
      doc: ~S"""
      Constant that multiplies the $L_2$ term, controlling regularization strength.
      `:alpha` must be a non-negative float i.e. in [0, inf).

      If `:alpha` is set to 0.0 the objective is the ordinary least squares regression.
      In this case, for numerical reasons, you should use `Scholar.Linear.LinearRegression` instead.
      """
    ],
    alpha_2: [
      type:
        {:or,
         [
           {:custom, Scholar.Options, :non_negative_number, []},
           {:list, {:custom, Scholar.Options, :non_negative_number, []}},
           {:custom, Scholar.Options, :weights, []}
         ]},
      default: 1.0e-6,
      doc: ~S"""
      Constant that multiplies the $L_2$ term, controlling regularization strength.
      `:alpha` must be a non-negative float i.e. in [0, inf).

      If `:alpha` is set to 0.0 the objective is the ordinary least squares regression.
      In this case, for numerical reasons, you should use `Scholar.Linear.LinearRegression` instead.
      """
    ],
    lambda: [
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
    ],
    lambda_1: [
      type:
        {:or,
         [
           {:custom, Scholar.Options, :non_negative_number, []},
           {:list, {:custom, Scholar.Options, :non_negative_number, []}},
           {:custom, Scholar.Options, :weights, []}
         ]},
      default: 1.0e-6,
      doc: ~S"""
      Constant that multiplies the $L_2$ term, controlling regularization strength.
      `:alpha` must be a non-negative float i.e. in [0, inf).

      If `:alpha` is set to 0.0 the objective is the ordinary least squares regression.
      In this case, for numerical reasons, you should use `Scholar.Linear.LinearRegression` instead.
      """
    ],
    lambda_2: [
      type:
        {:or,
         [
           {:custom, Scholar.Options, :non_negative_number, []},
           {:list, {:custom, Scholar.Options, :non_negative_number, []}},
           {:custom, Scholar.Options, :weights, []}
         ]},
      default: 1.0e-6,
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

    ## TODO: refactor this
    {alpha, opts} = Keyword.pop!(opts, :alpha)
    alpha = Nx.tensor(alpha, type: x_type) |> Nx.flatten()
    {alpha_1, opts} = Keyword.pop!(opts, :alpha_1)
    alpha_1 = Nx.tensor(alpha_1, type: x_type) |> Nx.flatten()
    {alpha_2, opts} = Keyword.pop!(opts, :alpha_2)
    alpha_2 = Nx.tensor(alpha_2, type: x_type) |> Nx.flatten()
    {lambda, opts} = Keyword.pop!(opts, :lambda)
    lambda = Nx.tensor(lambda, type: x_type) |> Nx.flatten()
    {lambda_1, opts} = Keyword.pop!(opts, :lambda_1)
    lambda_1 = Nx.tensor(lambda_1, type: x_type) |> Nx.flatten()
    {lambda_2, opts} = Keyword.pop!(opts, :lambda_2)
    lambda_2 = Nx.tensor(lambda_2, type: x_type) |> Nx.flatten()

    num_targets = if Nx.rank(y) == 1, do: 1, else: Nx.axis_size(y, 1)

    if Nx.size(alpha) not in [0, 1, num_targets] do
      raise ArgumentError,
            "expected number of targets be the same as number of penalties, got: #{inspect(num_targets)} != #{inspect(Nx.size(alpha))}"
    end

    {{coefficients, _rmse, iterations, has_converged}, _} =
      fit_n(x, y, sample_weights, alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2, 300)
    if Nx.to_number(has_converged) == 1 do
      IO.puts("Convergence after #{Nx.to_number(iterations)} iterations")
    end
    %__MODULE__{coefficients: coefficients}
  end

  defnp fit_n(x, y, sample_weights, alpha, lambda,
              alpha_1, alpha_2, lambda_1, lambda_2, iterations) do
    print_value(Nx.u8(0))
    xt_y = Nx.dot(Nx.transpose(x), y)
    {u, s, vh} = Nx.LinAlg.svd(x, full_matrices?: false)
    eigenvals = Nx.pow(s, 2)
    {n_samples, n_features} = Nx.shape(x)
    {coef, rmse} = update_coef(x, y, n_samples, n_features,
                               xt_y, u, vh, eigenvals,
                               alpha, lambda)

    {{final_coef, rmse, iter, has_converged}, _} =
      while {{coef, rmse, iter = 1, has_converged = Nx.u8(0)},
             {x, y,
              xt_y, u, s, vh, eigenvals,
              alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2,
              iterations}},
            iter < iterations and not has_converged do
        {coef_new, rmse} = update_coef(
                             x, y, n_samples, n_features,
                             xt_y, u, vh, eigenvals,
                             alpha, lambda)
        
        gamma = Nx.sum(alpha * eigenvals / (lambda + alpha * eigenvals))
        lambda = (gamma + 2 * lambda_1) / (Nx.sum(coef_new ** 2) + 2 * lambda_2)
        alpha = (n_samples - gamma + 2 * alpha_1) / (rmse + 2 * alpha_2)
        
        has_converged = check_convergence(coef_new, coef, 1.0e-10)
        
        {{coef_new, rmse, iter + 1, has_converged},
         {x, y,
          xt_y, u, s, vh, eigenvals,
          alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2,
          iterations}}
      end
  end

  defnp update_coef(
          x,
          y,
          _n_samples,
          _n_features,
          xt_y,
          u,
          vh,
          eigenvals,
          alpha,
          lambda
        ) do
    regularization = vh / (eigenvals + lambda / alpha)
    reg_transpose = Nx.dot(regularization, xt_y)
    coef = Nx.dot(Nx.transpose(vh), reg_transpose)

    error = y - Nx.dot(x, coef)
    squared_error = error ** 2
    rmse = Nx.sum(squared_error)

    {coef, rmse}
  end

  defnp check_convergence(coef_old, coef_new, tol) do
    Nx.less(Nx.sum(Nx.abs(coef_old - coef_new)), tol)
  end

  defn predict(%__MODULE__{coefficients: coeff} = _model, x) do
    Nx.dot(x, coeff)
  end
end
