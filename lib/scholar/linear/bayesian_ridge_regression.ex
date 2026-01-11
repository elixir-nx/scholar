defmodule Scholar.Linear.BayesianRidgeRegression do
  @moduledoc ~S"""
  Bayesian ridge regression: A fully probabilistic linear model with parameter regularization.

  In order to obtain a fully probabilistic linear model,
  we declare the precision parameter in the model: $\alpha$,
  This parameter describes the dispersion of the data around the mean.

  $$
  p(y | X, w, \alpha) = \mathcal{N}(y | Xw, \alpha^{-1})
  $$

  Where:
  * $X$ is an input data

  * $y$ is an input target

  * $w$ is the model weights matrix

  * $\alpha$ is the precision parameter of the target and $\alpha^{-1} = \sigma^{2}$, the variance.

  In order to obtain a fully probabilistic regularized linear model,
  we declare the distribution of the model weights matrix
  with it's corresponding precision parameter:

  $$
  p(w | \lambda) = \mathcal{N}(w, \lambda^{-1})
  $$

  Where $\lambda$ is the precision parameter of the weights matrix.

  Both $\alpha$ and $\lambda$ are choosen to have prior gamma distributions,
  controlled through hyperparameters $\alpha_1$, $\alpha_2$, $\lambda_1$, $\lambda_2$.
  These parameters are set by default to non-informative
  $\alpha_1 = \alpha_2 = \lambda_1 = \lambda_2 = 1^{-6}$.

  This model is similar to the classical ridge regression.
  Confusingly the classical ridge regression's $\alpha$ parameter is the Bayesian ridge's $\lambda$ parameter.

  Other than that, the differences between alorithms are:
  * The matrix weight regularization parameter is estimated from data,
  * The precision of the target is estimated.

  As such, Bayesian ridge is more flexible to the data at hand.
  These features come at higher computational cost.

  This implementation is ported from Python's scikit-learn.
  It uses the algorithm described in (Tipping, 2001)
  and regularization parameters are updated as by (MacKay, 1992).

  References:

  D. J. C. MacKay, Bayesian Interpolation, Computation and Neural Systems,
  Vol. 4, No. 3, 1992.

  M. E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine,
  Journal of Machine Learning Research, Vol. 1, 2001.

  Pedregosa et al., Scikit-learn: Machine Learning in Python,
  JMLR 12, pp. 2825-2830, 2011.
  """
  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Linear.LinearHelpers

  @derive {Nx.Container,
           containers: [
             :coefficients,
             :intercept,
             :alpha,
             :lambda,
             :sigma,
             :iterations,
             :has_converged,
             :scores
           ]}
  defstruct [
    :coefficients,
    :intercept,
    :alpha,
    :lambda,
    :sigma,
    :iterations,
    :has_converged,
    :scores
  ]

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
      type: {:custom, Scholar.Options, :weights, []},
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
    compute_scores?: [
      type: :boolean,
      default: false,
      doc: """
      If set to `true`, the log marginal likelihood will be computed
      at each iteration of the algorithm.
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

  @doc """
  Fits a Bayesian ridge model for sample inputs `x` and
  sample targets `y`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:coefficients` - Estimated coefficients for the linear regression problem.

    * `:intercept` - Independent term in the linear model.

    * `:alpha` - Estimated precision of the noise.

    * `:lambda` - Estimated precision of the weights.

    * `:sigma` - Estimated variance covariance matrix of weights with shape (n_features, n_features).

    * `:iterations` - How many times the optimization algorithm was computed.

    * `:has_converged` - Whether the coefficients converged during the optimization algorithm.

    * `:scores` - Value of the log marginal likelihood at each iteration during the optimization.

  ## Examples

      iex> x = Nx.tensor([[1], [2], [6], [8], [10]])
      iex> y = Nx.tensor([1, 2, 6, 8, 10])
      iex> model = Scholar.Linear.BayesianRidgeRegression.fit(x, y)
      iex> model.coefficients
      #Nx.Tensor<
        f32[1]
        [0.9932512044906616]
      >
      iex> model.intercept
      #Nx.Tensor<
        f32
        0.03644371032714844
      >
  """
  deftransform fit(x, y, opts \\ []) do
    {n_samples, _} = Nx.shape(x)
    y = LinearHelpers.validate_y_shape(y, n_samples, __MODULE__)

    opts = NimbleOptions.validate!(opts, @opts_schema)

    opts =
      [
        sample_weights_flag: opts[:sample_weights] != nil
      ] ++
        opts

    x_type = to_float_type(x)

    sample_weights = LinearHelpers.build_sample_weights(x, opts)

    # handle vector types
    # handle default alpha value, add eps to avoid division by 0
    eps = Nx.Constants.smallest_positive_normal(x_type)
    default_alpha = Nx.divide(1, Nx.add(Nx.variance(x), eps))
    alpha = Keyword.get(opts, :alpha_init, default_alpha)
    alpha = Nx.tensor(alpha, type: x_type)

    {lambda, opts} = Keyword.pop!(opts, :lambda_init)
    lambda = Nx.tensor(lambda, type: x_type)

    scores =
      Nx.tensor(:nan)
      |> Nx.broadcast({opts[:iterations] + 1})
      |> Nx.as_type(x_type)

    {coefficients, intercept, alpha, lambda, iterations, has_converged, scores, sigma} =
      fit_n(x, y, alpha, lambda, sample_weights, scores, opts)

    scores =
      if opts[:compute_scores?] do
        scores
      else
        nil
      end

    %__MODULE__{
      coefficients: coefficients,
      intercept: intercept,
      alpha: alpha,
      lambda: lambda,
      sigma: sigma,
      iterations: iterations,
      has_converged: has_converged,
      scores: scores
    }
  end

  defnp fit_n(x, y, alpha, lambda, sample_weights, scores, opts) do
    x = to_float(x)
    y = to_float(y)

    {x_offset, y_offset} =
      if opts[:fit_intercept?] do
        LinearHelpers.preprocess_data(x, y, sample_weights, opts)
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
        LinearHelpers.rescale(x, y, sample_weights)
      else
        {x, y}
      end

    alpha_1 = opts[:alpha_1]
    alpha_2 = opts[:alpha_2]
    lambda_1 = opts[:lambda_1]
    lambda_2 = opts[:lambda_2]

    iterations = opts[:iterations]

    xt_y = Nx.dot(x, [0], y, [0])
    {u, s, vh} = Nx.LinAlg.svd(x, full_matrices?: false)
    eigenvals = s ** 2
    {n_samples, n_features} = Nx.shape(x)
    {coef, rmse} = update_coef(x, y, n_samples, n_features, xt_y, u, vh, eigenvals, alpha, lambda)

    {{coef, alpha, lambda, _rmse, iter, has_converged, scores}, _} =
      while {{coef, rmse, alpha, lambda, iter = Nx.u64(0), has_converged = Nx.u8(0),
              scores = scores},
             {x, y, xt_y, u, s, vh, eigenvals, alpha_1, alpha_2, lambda_1, lambda_2, iterations}},
            iter <= iterations and not has_converged do
        scores =
          if opts[:compute_scores?] do
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

            Nx.put_slice(scores, [iter], Nx.new_axis(new_score, -1))
          else
            scores
          end

        gamma = Nx.sum(alpha * eigenvals / (lambda + alpha * eigenvals))
        lambda = (gamma + 2 * lambda_1) / (Nx.sum(coef ** 2) + 2 * lambda_2)
        alpha = (n_samples - gamma + 2 * alpha_1) / (rmse + 2 * alpha_2)

        {coef_new, rmse} =
          update_coef(x, y, n_samples, n_features, xt_y, u, vh, eigenvals, alpha, lambda)

        has_converged = Nx.sum(Nx.abs(coef - coef_new)) < 1.0e-8

        {{coef_new, alpha, lambda, rmse, iter + 1, has_converged, scores},
         {x, y, xt_y, u, s, vh, eigenvals, alpha_1, alpha_2, lambda_1, lambda_2, iterations}}
      end

    intercept = LinearHelpers.set_intercept(coef, x_offset, y_offset, opts[:fit_intercept?])
    scaled_sigma = Nx.dot(vh, [0], vh / Nx.new_axis(eigenvals + lambda / alpha, -1), [0])
    sigma = scaled_sigma / alpha
    {coef, intercept, alpha, lambda, iter, has_converged, scores, sigma}
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

    coef =
      if n_samples > n_features do
        regularization = vh / Nx.new_axis(scaled_eigens, -1)
        reg_transpose = Nx.dot(regularization, xt_y)
        Nx.dot(vh, [0], reg_transpose, [0])
      else
        regularization = u / scaled_eigens
        reg_transpose = Nx.dot(regularization, Nx.dot(u, [0], y, [0]))
        Nx.dot(x, [0], reg_transpose, [0])
      end

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
    logdet_sigma =
      if n_samples > n_features do
        -1 * Nx.sum(Nx.log(lambda + alpha * eigenvals))
      else
        broad_lambda = Nx.broadcast(lambda, {n_samples})
        -1 * Nx.sum(Nx.log(broad_lambda + alpha * eigenvals))
      end

    score_lambda = lambda_1 * Nx.log(lambda) - lambda_2 * lambda
    score_alpha = alpha_1 * Nx.log(alpha) - alpha_2 * alpha

    score_parameters =
      n_features * Nx.log(lambda) + n_samples * Nx.log(alpha) - alpha * rmse -
        lambda * Nx.sum(coef ** 2)

    score =
      0.5 * (score_parameters + logdet_sigma - n_samples * Nx.log(2 * Nx.Constants.pi()))

    score_alpha + score_lambda + score
  end

  @doc """
  Makes predictions with the given `model` on input `x`.

  Output predictions have shape `{n_samples}` when train target is shaped either `{n_samples}` or `{n_samples, 1}`.

  ## Examples

      iex> x = Nx.tensor([[1], [2], [6], [8], [10]])
      iex> y = Nx.tensor([1, 2, 6, 8, 10])
      iex> model = Scholar.Linear.BayesianRidgeRegression.fit(x, y)
      iex> Scholar.Linear.BayesianRidgeRegression.predict(model, Nx.tensor([[1], [3], [4]]))
      Nx.tensor(
        [1.02969491481781, 3.0161972045898438, 4.009448528289795]  
      )  
  """
  deftransform predict(%__MODULE__{coefficients: coeff, intercept: intercept} = _model, x) do
    predict_n(coeff, intercept, x)
  end

  defnp predict_n(coeff, intercept, x), do: Nx.dot(x, [-1], coeff, [-1]) + intercept
end
