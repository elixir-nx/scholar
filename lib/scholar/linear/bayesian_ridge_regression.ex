defmodule Scholar.Linear.BayesianRidgeRegression do
  require Nx
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:coefficients]}
  defstruct [:coefficients]

  opts = [
    iterations: [
      type: :pos_integer,
      default: 300,
      doc: """
      Maximum number of iterations before stopping the fitting algorithm.
      The number of iterations may be lower is parameters converge.
      """
    ],    
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
    alpha_init: [
      type:
        {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0,
      doc: ~S"""
      The initial value for alpha. This parameter influences the precision of the noise.
      `:alpha` must be a non-negative float i.e. in [0, inf).
      """
    ],
    lambda_init: [
      type:
        {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0,
      doc: ~S"""
      The initial value for lambda. This parameter influences the precision of the weights.
      `:lambda` must be a non-negative float i.e. in [0, inf).      
      """
    ],
    alpha_1: [
      type:
        {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-6,
      doc: ~S"""
      Hyper-parameter : shape parameter for the Gamma distribution prior      
      over the alpha parameter.
      """
    ],    
    alpha_2: [
      type:
        {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-6,
      doc: ~S"""
      Hyper-parameter : inverse scale (rate) parameter for the Gamma distribution prior
      over the alpha parameter.
      """
    ],
    lambda_1: [
      type:
        {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-6,
      doc: ~S"""
      Hyper-parameter : shape parameter for the Gamma distribution prior
      over the lambda parameter.            
      """
    ],
    lambda_2: [
      type:
        {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-6,
      doc: ~S"""
      Hyper-parameter : inverse scale (rate) parameter for the Gamma distribution prior      
      over the lambda parameter.            
      """
    ],    
    eps: [
      type: :float,
      default: 1.0e-8,
      doc:
        "The convergence tolerance. When `Nx.sum(Nx.abs(coef - coef_new)) < :eps`, the algorithm is considered to have converged."
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

    IO.inspect(opts)
    lambda = Keyword.get(opts, :lambda_init, 1 / Nx.variance(y))
    opts = Keyword.put(opts, :lambda_init, lambda)
    IO.inspect(opts)

    num_targets = if Nx.rank(y) == 1, do: 1, else: Nx.axis_size(y, 1)

    {{coefficients, _rmse, iterations, has_converged}, _} =
      fit_n(x, y, sample_weights, opts)

    if Nx.to_number(has_converged) == 1 do
      IO.puts("Convergence after #{Nx.to_number(iterations)} iterations")
    end
    %__MODULE__{coefficients: coefficients}
  end

  defnp fit_n(x, y, sample_weights, opts) do
    alpha = opts[:alpha_init]
    alpha_1 = opts[:alpha_1]
    alpha_2 = opts[:alpha_2]
    lambda = opts[:lambda_init]
    lambda = opts[:lambda_init]
               
    lambda_1 = opts[:lambda_1]
    lambda_2 = opts[:lambda_2]
    iterations = opts[:iterations]    
    
    xt_y = Nx.dot(Nx.transpose(x), y)
    {u, s, vh} = Nx.LinAlg.svd(x, full_matrices?: false)
    eigenvals = Nx.pow(s, 2)
    {n_samples, n_features} = Nx.shape(x)
    {coef, rmse} = update_coef(x, y, n_samples, n_features,
                               xt_y, u, vh, eigenvals,
                               alpha, lambda)

    while {{coef, rmse, iter = 0, has_converged = Nx.u8(0)},
           {x, y,
            xt_y, u, s, vh, eigenvals,
            alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2,
            iterations}},
      iter < iterations and not has_converged do

      # gamma = Nx.sum(alpha * eigenvals / (lambda + alpha * eigenvals))
      gamma =
        Nx.multiply(alpha, eigenvals)
        |> Nx.divide(Nx.multiply(lambda + alpha, eigenvals))
        |> Nx.sum()      
      lambda = (gamma + 2 * lambda_1) / (Nx.sum(coef ** 2) + 2 * lambda_2)
      alpha = (n_samples - gamma + 2 * alpha_1) / (rmse + 2 * alpha_2)
      
      {coef_new, rmse} = update_coef(
        x, y, n_samples, n_features,
        xt_y, u, vh, eigenvals,
        alpha, lambda)
      
      has_converged = Nx.sum(Nx.abs(coef - coef_new)) < 1.0e-8
      {{coef_new, rmse, iter + 1, has_converged},
       {x, y,
        xt_y, u, s, vh, eigenvals,
        alpha, lambda, alpha_1, alpha_2, lambda_1, lambda_2,
        iterations}}
    end
  end

  defn update_coef(
    x,
    y,
    n_samples,
    n_features,
    xt_y,
    u,
    vh,
    eigenvals,
    alpha,
    lambda
  ) do
    scaled_eigens = eigenvals + lambda / alpha
    {n_eigens} = Nx.shape(scaled_eigens)
    regularization = vh / Nx.new_axis(scaled_eigens, -1) 
    reg_transpose = Nx.dot(regularization, xt_y)
    coef = Nx.dot(Nx.transpose(vh), reg_transpose)

    error = y - Nx.dot(x, coef)
    squared_error = error ** 2
    rmse = Nx.sum(squared_error)

    {coef, rmse}
  end

  defn predict(%__MODULE__{coefficients: coeff} = _model, x) do
    Nx.dot(x, coeff)
  end
end
