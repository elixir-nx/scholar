defmodule Scholar.Linear.BayesianRidgeRegression do
  require Nx
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container,
           containers: [:coefficients, :intercept, :alpha, :lambda, :rmse, :iterations, :scores]}
  defstruct [:coefficients, :intercept, :alpha, :lambda, :rmse, :iterations, :scores]

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
      type: {:custom, Scholar.Options, :non_negative_number, []},
      doc: ~S"""
      The initial value for alpha. This parameter influences the precision of the noise.
      `:alpha` must be a non-negative float i.e. in [0, inf).
      Defaults to 1/Var(y).      
      """
    ],
    lambda_init: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0,
      doc: ~S"""
      The initial value for lambda. This parameter influences the precision of the weights.
      `:lambda` must be a non-negative float i.e. in [0, inf).
      Defaults to 1.
      """
    ],
    alpha_1: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-6,
      doc: ~S"""
      Hyper-parameter : shape parameter for the Gamma distribution prior      
      over the alpha parameter.
      """
    ],
    alpha_2: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-6,
      doc: ~S"""
      Hyper-parameter : inverse scale (rate) parameter for the Gamma distribution prior
      over the alpha parameter.
      """
    ],
    lambda_1: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-6,
      doc: ~S"""
      Hyper-parameter : shape parameter for the Gamma distribution prior
      over the lambda parameter.            
      """
    ],
    lambda_2: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
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

    opts =
      [
        sample_weights_flag: opts[:sample_weights] != nil
      ] ++
        opts

    {sample_weights, opts} = Keyword.pop(opts, :sample_weights, 1.0)
    x_type = to_float_type(x)

    sample_weights =
      if Nx.is_tensor(sample_weights),
        do: Nx.as_type(sample_weights, x_type),
        else: Nx.tensor(sample_weights, type: x_type)

    # handle vector types
    # handle default alpha value, add eps to avoid division by 0
    eps = Nx.Constants.smallest_positive_normal(x_type)
    default_alpha = Nx.divide(1, Nx.add(Nx.variance(x), eps))
    alpha = Keyword.get(opts, :alpha_init, default_alpha)
    alpha = Nx.tensor(alpha, type: x_type)
    opts = Keyword.put(opts, :alpha_init, alpha)

    {lambda, opts} = Keyword.pop!(opts, :lambda_init)
    lambda = Nx.tensor(lambda, type: x_type)
    opts = Keyword.put(opts, :lambda_init, lambda)
    zeros_list = for k <- 0..opts[:iterations], do: 0
    scores = Nx.tensor(zeros_list, type: x_type)
    IO.inspect(scores)

    {coefficients, intercept, alpha, lambda, rmse, iterations, has_converged, scores} =
      fit_n(x, y, sample_weights, scores, opts)
    iterations = Nx.to_number(iterations)
    scores = scores
    |> Nx.to_list()
    |> Enum.take(iterations)

    if Nx.to_number(has_converged) == 1 do
      IO.puts("Convergence after #{Nx.to_number(iterations)} iterations")
    end

    %__MODULE__{
      coefficients: coefficients,
      intercept: intercept,
      alpha: Nx.to_number(alpha),
      lambda: Nx.to_number(lambda),
      rmse: Nx.to_number(rmse),
      iterations: iterations,
      scores: scores
    }
  end

  defnp fit_n(x, y, sample_weights, scores, opts) do
    x = to_float(x)
    y = to_float(y)

    {x_offset, y_offset} =
      if opts[:fit_intercept?] do
        preprocess_data(x, y, sample_weights, opts)
      else
        x_offset_shape = Nx.axis_size(x, 1)
        y_reshaped = if Nx.rank(y) > 1, do: y, else: Nx.reshape(y, {:auto, 1})
        y_offset_shape = Nx.axis_size(y_reshaped, 1)

        {Nx.broadcast(Nx.tensor(0.0, type: Nx.type(x)), {x_offset_shape}),
         Nx.broadcast(Nx.tensor(0.0, type: Nx.type(y)), {y_offset_shape})}
      end

    {x, y} = {x - x_offset, y - y_offset}

    {x, y} =
      if opts[:sample_weights_flag] do
        rescale(x, y, sample_weights)
      else
        {x, y}
      end

    alpha = opts[:alpha_init]
    lambda = opts[:lambda_init]

    alpha_1 = opts[:alpha_1]
    alpha_2 = opts[:alpha_2]
    lambda_1 = opts[:lambda_1]
    lambda_2 = opts[:lambda_2]

    iterations = opts[:iterations]

    xt_y = Nx.dot(Nx.transpose(x), y)
    {u, s, vh} = Nx.LinAlg.svd(x, full_matrices?: false)
    eigenvals = Nx.pow(s, 2)
    {n_samples, n_features} = Nx.shape(x)
    {coef, rmse} = update_coef(x, y, n_samples, n_features, xt_y, u, vh, eigenvals, alpha, lambda)

    {{coef, alpha, lambda, rmse, iter, has_converged, scores}, _} =
      while {{coef, rmse, alpha, lambda, iter = 0, has_converged = Nx.u8(0), scores = scores},
             {x, y, xt_y, u, s, vh, eigenvals, alpha_1, alpha_2, lambda_1, lambda_2, iterations}},
            iter <= iterations and not has_converged do
        new_score =
          log_marginal_likelihood(
            coef,
            rmse,
            n_samples,
            n_features,
            eigenvals,
            alpha,
            lambda,
            alpha_1,
            alpha_2,
            lambda_1,
            lambda_2
          )
        scores = Nx.put_slice(scores, [iter], Nx.new_axis(new_score, -1))
        
        gamma = Nx.sum(alpha * eigenvals / (lambda + alpha * eigenvals))
        lambda = (gamma + 2 * lambda_1) / (Nx.sum(coef ** 2) + 2 * lambda_2)
        alpha = (n_samples - gamma + 2 * alpha_1) / (rmse + 2 * alpha_2)

        {coef_new, rmse} =
          update_coef(x, y, n_samples, n_features, xt_y, u, vh, eigenvals, alpha, lambda)

        has_converged = Nx.sum(Nx.abs(coef - coef_new)) < 1.0e-8

        {{coef_new, alpha, lambda, rmse, iter + 1, has_converged, scores},
         {x, y, xt_y, u, s, vh, eigenvals, alpha_1, alpha_2, lambda_1, lambda_2, iterations}}
      end

    intercept = set_intercept(coef, x_offset, y_offset, opts[:fit_intercept?])
    {coef, intercept, alpha, lambda, rmse, iter, has_converged, scores}
  end

  defnp update_coef(
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
    regularization = vh / Nx.new_axis(scaled_eigens, -1)
    reg_transpose = Nx.dot(regularization, xt_y)
    coef = Nx.dot(Nx.transpose(vh), reg_transpose)

    error = y - Nx.dot(x, coef)
    squared_error = error ** 2
    rmse = Nx.sum(squared_error)

    {coef, rmse}
  end

  defnp log_marginal_likelihood(
          coef,
          rmse,
          n_samples,
          n_features,
          eigenvals,
          alpha,
          lambda,
          alpha_1,
          alpha_2,
          lambda_1,
          lambda_2
        ) do
    logdet_sigma = -1 * Nx.sum(Nx.log(lambda + alpha * eigenvals))
    score_lambda = lambda_1 * Nx.log(lambda) - lambda_2 * lambda
    score_alpha = alpha_1 * Nx.log(alpha) - alpha_2 * alpha

    score_parameters =
      n_features * Nx.log(lambda) + n_samples * Nx.log(alpha) - alpha * rmse -
        lambda * Nx.sum(coef ** 2)

    score =
      0.5 * (score_parameters + logdet_sigma - n_samples * Nx.log(2 * Nx.Constants.pi()))

    score_alpha + score_lambda + score
  end

  defn predict(%__MODULE__{coefficients: coeff, intercept: intercept} = _model, x) do
    Nx.dot(x, [-1], coeff, [-1]) + intercept
  end

  # Implements sample weighting by rescaling inputs and
  # targets by sqrt(sample_weight).
  defnp rescale(x, y, sample_weights) do
    case Nx.shape(sample_weights) do
      {} = scalar ->
        scalar = Nx.sqrt(scalar)
        {scalar * x, scalar * y}

      _ ->
        scale = sample_weights |> Nx.sqrt() |> Nx.make_diagonal()
        {Nx.dot(scale, x), Nx.dot(scale, y)}
    end
  end

  defnp set_intercept(coeff, x_offset, y_offset, fit_intercept?) do
    if fit_intercept? do
      y_offset - Nx.dot(x_offset, coeff)
    else
      Nx.tensor(0.0, type: Nx.type(coeff))
    end
  end

  defnp preprocess_data(x, y, sample_weights, opts) do
    if opts[:sample_weights_flag],
      do:
        {Nx.weighted_mean(x, sample_weights, axes: [0]),
         Nx.weighted_mean(y, sample_weights, axes: [0])},
      else: {Nx.mean(x, axes: [0]), Nx.mean(y, axes: [0])}
  end
end
