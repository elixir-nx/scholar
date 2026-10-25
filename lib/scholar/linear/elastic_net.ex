defmodule Scholar.Linear.ElasticNet do
  @moduledoc ~S"""
  Linear regression with combined $L_1$ and $L_2$ regularization.

  Minimizes the objective function:
  $$
  \frac{1}{2n}||y - Xw||^2\_2 + \alpha\rho||w||\_1 + \frac{\alpha(1 - \rho)}{2}||w||^2\_2
  $$

  Where:
  * $X$ is an input data

  * $y$ is an input target

  * $w$ is the model weights matrix

  * $n$ is the number of samples

  * $\alpha$ is the parameter that controls the level of regularization

  * $\rho$ is the `:l1_ratio`, the share of the $L_1$ term in the penalty

  With `l1_ratio: 1.0` the penalty is the $L_1$ norm alone, which is the Lasso
  (see `Scholar.Linear.Lasso`). The $L_1$ term drives coefficients to exactly
  zero, so the fitted model is sparse.

  The model is fitted with cyclic coordinate descent, the same algorithm as
  scikit-learn's `ElasticNet`. Each iteration updates every coefficient once,
  in order. When the largest update of an iteration is smaller than `:tol`
  relative to the largest coefficient, the duality gap is computed, and the
  fit stops once the gap is below `:tol` times $||y||^2\_2$.

  Time complexity is $O(I \cdot N \cdot K)$, where $I$ is the number of
  iterations, $N$ is the number of samples and $K$ is the number of features.
  """
  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Linear.LinearHelpers

  @derive {Nx.Container, containers: [:coefficients, :intercept, :dual_gap, :num_iterations]}
  defstruct [:coefficients, :intercept, :dual_gap, :num_iterations]

  opts = [
    sample_weights: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: """
      The non-negative weights for each observation. If not provided,
      all observations are assigned equal weight. The weights are rescaled
      to sum to the number of samples, so only their ratios matter.
      """
    ],
    alpha: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0,
      doc: ~S"""
      Constant that multiplies the penalty terms, controlling regularization strength.
      `:alpha` must be a non-negative float i.e. in [0, inf).

      If `:alpha` is set to 0.0 the objective is the ordinary least squares regression.
      In this case, for numerical reasons, you should use `Scholar.Linear.LinearRegression` instead.
      """
    ],
    l1_ratio: [
      type: {:custom, __MODULE__, :l1_ratio, []},
      default: 0.5,
      doc: ~S"""
      The mixing parameter $\rho$, with `0.0 <= l1_ratio <= 1.0`. For `l1_ratio: 0.0`
      the penalty is the $L_2$ norm alone, for `l1_ratio: 1.0` it is the $L_1$ norm
      alone, and in between it is a combination of the two.

      With `l1_ratio: 0.0` the duality gap no longer detects convergence, so the fit
      runs until `:max_iterations`, as it does in scikit-learn. In this case you
      should use `Scholar.Linear.RidgeRegression` instead.
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
    positive?: [
      type: :boolean,
      default: false,
      doc: "If set to `true`, the coefficients are constrained to be non-negative."
    ],
    max_iterations: [
      type: :pos_integer,
      default: 1000,
      doc: "Maximum number of coordinate descent iterations to perform."
    ],
    tol: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-4,
      doc: ~S"""
      Tolerance for the optimization. Once the largest coefficient update of an
      iteration is smaller than `:tol` times the largest coefficient, the duality
      gap is checked, and the fit stops when the gap is smaller than `:tol`
      times $||y||^2\_2$.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits an elastic net model for sample inputs `x` and sample targets `y`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:coefficients` - Estimated coefficients for the linear regression problem,
      of shape `{num_features}` for targets of shape `{num_samples}`, and
      `{num_targets, num_features}` for targets of shape `{num_samples, num_targets}`.

    * `:intercept` - Independent term in the linear model, one per target.
      It is a scalar `0.0` when `:fit_intercept?` is `false`.

    * `:dual_gap` - Duality gap at the end of the optimization, divided by the
      number of samples so it is on the scale of the objective function.
      One per target.

    * `:num_iterations` - Number of coordinate descent iterations performed,
      one per target. When it equals `:max_iterations`, the fit stopped before
      reaching the tolerance.

  Each target is fitted on its own, the same as fitting the model once per
  column of `y`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> Scholar.Linear.ElasticNet.fit(x, y, alpha: 0.1)
      %Scholar.Linear.ElasticNet{
        coefficients: Nx.tensor(
          [-0.4430642, -0.7074076]
        ),
        intercept: Nx.tensor(
          5.7753325
        ),
        dual_gap: Nx.tensor(
          2.1226704e-5
        ),
        num_iterations: Nx.tensor(
          17,
          type: :u32
        )
      }
  """
  deftransform fit(x, y, opts \\ []) do
    {num_samples, _} = Nx.shape(x)
    y = LinearHelpers.validate_multi_target_y_shape(y, num_samples, __MODULE__)

    opts = NimbleOptions.validate!(opts, @opts_schema)

    type = Nx.Type.merge(to_float_type(x), to_float_type(y))
    sample_weights = validate_weights(opts[:sample_weights], num_samples, type: type)

    # Penalties on the scale of 1/2 ||y - Xw||^2, the objective that coordinate
    # descent minimizes. They stay Elixir numbers to keep their f64 precision.
    l1_reg = opts[:alpha] * opts[:l1_ratio] * num_samples
    l2_reg = opts[:alpha] * (1.0 - opts[:l1_ratio]) * num_samples

    fit_n(x, y, sample_weights,
      type: type,
      l1_reg: l1_reg,
      l2_reg: l2_reg,
      tol: opts[:tol],
      max_iterations: opts[:max_iterations],
      fit_intercept?: opts[:fit_intercept?],
      positive?: opts[:positive?]
    )
  end

  # Shared with `Scholar.Linear.Lasso`, which is this model with `l1_ratio: 1.0`.
  #
  # `sample_weights` is a scalar when the user gave no weights, and a tensor
  # of shape {num_samples} otherwise.
  @doc false
  defn fit_n(x, y, sample_weights, opts) do
    x = Nx.as_type(x, opts[:type])
    y = Nx.as_type(y, opts[:type])
    {num_samples, num_features} = Nx.shape(x)
    weighted? = Nx.rank(sample_weights) == 1

    single_target? = Nx.rank(y) == 1
    y = if single_target?, do: Nx.new_axis(y, 1), else: y
    num_targets = Nx.axis_size(y, 1)

    {x_offset, y_offset} =
      if opts[:fit_intercept?] do
        if weighted? do
          {Nx.weighted_mean(x, sample_weights, axes: [0]),
           Nx.weighted_mean(y, sample_weights, axes: [0])}
        else
          {Nx.mean(x, axes: [0]), Nx.mean(y, axes: [0])}
        end
      else
        {Nx.broadcast(Nx.tensor(0.0, type: opts[:type]), {num_features}),
         Nx.broadcast(Nx.tensor(0.0, type: opts[:type]), {num_targets})}
      end

    x = x - x_offset
    y = y - y_offset

    # Same as scikit-learn: the weights are rescaled to sum to num_samples,
    # so the penalties keep their scale, and the centered data is multiplied
    # by their square root.
    {x, y} =
      if weighted? do
        sample_weights = sample_weights * num_samples / Nx.sum(sample_weights)
        LinearHelpers.rescale(x, y, sample_weights)
      else
        {x, y}
      end

    {coefficients, dual_gap, num_iterations} = coordinate_descent(x, y, opts)

    intercept =
      LinearHelpers.set_intercept(coefficients, x_offset, y_offset, opts[:fit_intercept?])

    dual_gap = dual_gap / num_samples

    if single_target? do
      %__MODULE__{
        coefficients: Nx.squeeze(coefficients, axes: [0]),
        intercept: Nx.squeeze(intercept),
        dual_gap: Nx.squeeze(dual_gap, axes: [0]),
        num_iterations: Nx.squeeze(num_iterations, axes: [0])
      }
    else
      %__MODULE__{
        coefficients: coefficients,
        intercept: intercept,
        dual_gap: dual_gap,
        num_iterations: num_iterations
      }
    end
  end

  # Cyclic coordinate descent for each target k
  #
  #     1/2 ||y_k - X w_k||^2 + l1_reg ||w_k||_1 + l2_reg / 2 ||w_k||^2
  #
  # following scikit-learn's `enet_coordinate_descent`, so the iterates,
  # the stopping rule and the number of iterations are the same.
  #
  # The targets are solved together, one column each. scikit-learn fits them
  # one at a time, so a target that converges is frozen: its coefficients,
  # duality gap and iteration count stop changing while the others go on.
  defnp coordinate_descent(x, y, opts) do
    type = Nx.type(x)
    num_features = Nx.axis_size(x, 1)
    num_targets = Nx.axis_size(y, 1)
    l1_reg = opts[:l1_reg]
    l2_reg = opts[:l2_reg]
    max_iterations = opts[:max_iterations]
    d_w_tol = opts[:tol]

    xt = Nx.transpose(x)
    norm_cols = Nx.sum(xt * xt, axes: [1])
    gap_tol = d_w_tol * Nx.sum(y * y, axes: [0])

    zeros = Nx.broadcast(Nx.tensor(0.0, type: type), {num_targets})
    w = Nx.broadcast(Nx.tensor(0.0, type: type), {num_targets, num_features})
    gap = gap_tol + 1.0
    num_iterations = Nx.broadcast(Nx.u32(0), {num_targets})
    active = Nx.broadcast(Nx.u8(1), {num_targets})

    {{w, gap, num_iterations, _residual, _active, _iteration}, _} =
      while {{w, gap, num_iterations, residual = y, active, iteration = Nx.u32(0)},
             {xt, y, norm_cols, gap_tol, zeros}},
            iteration < max_iterations and Nx.any(active) do
        {{w, residual, w_max, d_w_max}, _} =
          while {{w, residual, w_max = zeros, d_w_max = zeros}, {xt, norm_cols, active}},
                j <- 0..(num_features - 1) do
            x_j = xt[j]
            norm_j = norm_cols[j]
            w_j = w |> Nx.slice_along_axis(j, 1, axis: 1) |> Nx.squeeze(axes: [1])

            # Residual without the contribution of feature j.
            residual = residual + Nx.outer(x_j, w_j)
            tmp = Nx.dot(x_j, [0], residual, [0])

            updated = Nx.sign(tmp) * Nx.max(Nx.abs(tmp) - l1_reg, 0) / (norm_j + l2_reg)

            updated =
              if opts[:positive?] do
                Nx.select(tmp < 0, 0.0, updated)
              else
                updated
              end

            # A zero column carries no information, its coefficient stays at
            # zero, and a target that already converged keeps its coefficients.
            new_w_j = Nx.select(norm_j == 0 or not active, w_j, updated)

            residual = residual - Nx.outer(x_j, new_w_j)
            w = Nx.put_slice(w, [0, j], Nx.new_axis(new_w_j, 1))

            {{w, residual, Nx.max(w_max, Nx.abs(new_w_j)),
              Nx.max(d_w_max, Nx.abs(new_w_j - w_j))}, {xt, norm_cols, active}}
          end

        check_gap? =
          active and
            (w_max == 0 or d_w_max / w_max < d_w_tol or iteration == max_iterations - 1)

        gap =
          if Nx.any(check_gap?) do
            new_gap = duality_gap(xt, y, w, residual, l1_reg, l2_reg, positive?: opts[:positive?])
            Nx.select(check_gap?, new_gap, gap)
          else
            gap
          end

        num_iterations = Nx.select(active, iteration + 1, num_iterations)
        active = active and not (check_gap? and gap < gap_tol)

        {{w, gap, num_iterations, residual, active, iteration + 1},
         {xt, y, norm_cols, gap_tol, zeros}}
      end

    {w, gap, num_iterations}
  end

  # Duality gap of each target, for `w` of shape {num_targets, num_features}
  # and `y` and `residual` of shape {num_samples, num_targets}.
  defnp duality_gap(xt, y, w, residual, l1_reg, l2_reg, opts) do
    xta = Nx.dot(xt, residual) - l2_reg * Nx.transpose(w)

    # With non-negative coefficients the dual constraint is one-sided.
    dual_norm_xta =
      if opts[:positive?] do
        Nx.reduce_max(xta, axes: [0])
      else
        Nx.reduce_max(Nx.abs(xta), axes: [0])
      end

    r_norm2 = Nx.sum(residual * residual, axes: [0])
    w_norm2 = Nx.sum(w * w, axes: [1])

    scale? = dual_norm_xta > l1_reg
    const = Nx.select(scale?, l1_reg / dual_norm_xta, 1.0)
    gap = Nx.select(scale?, 0.5 * (r_norm2 + r_norm2 * const ** 2), r_norm2)

    gap + l1_reg * Nx.sum(Nx.abs(w), axes: [1]) - const * Nx.sum(residual * y, axes: [0]) +
      0.5 * l2_reg * (1 + const ** 2) * w_norm2
  end

  @doc """
  Makes predictions with the given `model` on input `x`.

  Output predictions have shape `{num_samples}` when the model was fitted on
  targets of shape `{num_samples}` or `{num_samples, 1}`, and
  `{num_samples, num_targets}` otherwise.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> model = Scholar.Linear.ElasticNet.fit(x, y, alpha: 0.1)
      iex> Scholar.Linear.ElasticNet.predict(model, Nx.tensor([[2.0, 1.0]]))
      Nx.tensor(
        [4.1817966]
      )
  """
  defn predict(%__MODULE__{coefficients: coeff, intercept: intercept} = _model, x) do
    Nx.dot(x, [-1], coeff, [-1]) + intercept
  end

  @doc false
  def l1_ratio(value) do
    if is_number(value) and value >= 0 and value <= 1 do
      {:ok, value}
    else
      {:error, "expected :l1_ratio to be a number in the range [0, 1], got: #{inspect(value)}"}
    end
  end
end
