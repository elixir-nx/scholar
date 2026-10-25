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

    * `:coefficients` - Estimated coefficients for the linear regression problem.

    * `:intercept` - Independent term in the linear model.

    * `:dual_gap` - Duality gap at the end of the optimization, divided by the
      number of samples so it is on the scale of the objective function.

    * `:num_iterations` - Number of coordinate descent iterations performed.
      When it equals `:max_iterations`, the fit stopped before reaching the
      tolerance.

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
          2.1224221e-5
        ),
        num_iterations: Nx.tensor(
          17,
          type: :u32
        )
      }
  """
  deftransform fit(x, y, opts \\ []) do
    {num_samples, _} = Nx.shape(x)
    y = LinearHelpers.validate_y_shape(y, num_samples, __MODULE__)

    opts = NimbleOptions.validate!(opts, @opts_schema)

    type = Nx.Type.merge(to_float_type(x), to_float_type(y))

    # Penalties on the scale of 1/2 ||y - Xw||^2, the objective that coordinate
    # descent minimizes. They stay Elixir numbers to keep their f64 precision.
    l1_reg = opts[:alpha] * opts[:l1_ratio] * num_samples
    l2_reg = opts[:alpha] * (1.0 - opts[:l1_ratio]) * num_samples

    fit_n(x, y,
      type: type,
      l1_reg: l1_reg,
      l2_reg: l2_reg,
      tol: opts[:tol],
      max_iterations: opts[:max_iterations],
      fit_intercept?: opts[:fit_intercept?]
    )
  end

  # Shared with `Scholar.Linear.Lasso`, which is this model with `l1_ratio: 1.0`.
  @doc false
  defn fit_n(x, y, opts) do
    x = Nx.as_type(x, opts[:type])
    y = Nx.as_type(y, opts[:type])
    {num_samples, num_features} = Nx.shape(x)

    {x_offset, y_offset} =
      if opts[:fit_intercept?] do
        {Nx.mean(x, axes: [0]), Nx.mean(y)}
      else
        {Nx.broadcast(Nx.tensor(0.0, type: opts[:type]), {num_features}),
         Nx.tensor(0.0, type: opts[:type])}
      end

    x = x - x_offset
    y = y - y_offset

    {coefficients, dual_gap, num_iterations} = coordinate_descent(x, y, opts)

    intercept =
      LinearHelpers.set_intercept(coefficients, x_offset, y_offset, opts[:fit_intercept?])

    %__MODULE__{
      coefficients: coefficients,
      intercept: intercept,
      dual_gap: dual_gap / num_samples,
      num_iterations: num_iterations
    }
  end

  # Cyclic coordinate descent for
  #
  #     1/2 ||y - Xw||^2 + l1_reg ||w||_1 + l2_reg / 2 ||w||^2
  #
  # following scikit-learn's `enet_coordinate_descent`, so the iterates,
  # the stopping rule and the number of iterations are the same.
  defnp coordinate_descent(x, y, opts) do
    type = Nx.type(x)
    num_features = Nx.axis_size(x, 1)
    l1_reg = opts[:l1_reg]
    l2_reg = opts[:l2_reg]
    max_iterations = opts[:max_iterations]
    d_w_tol = opts[:tol]

    xt = Nx.transpose(x)
    norm_cols = Nx.sum(xt * xt, axes: [1])
    gap_tol = d_w_tol * Nx.dot(y, y)

    w = Nx.broadcast(Nx.tensor(0.0, type: type), {num_features})
    residual = y
    gap = gap_tol + 1.0

    {{w, _residual, gap, iteration, _converged?}, _} =
      while {{w, residual, gap, iteration = Nx.u32(0), converged? = Nx.u8(0)},
             {xt, y, norm_cols, gap_tol}},
            iteration < max_iterations and not converged? do
        {{w, residual, w_max, d_w_max}, _} =
          while {{w, residual, w_max = Nx.tensor(0.0, type: type),
                  d_w_max = Nx.tensor(0.0, type: type)}, {xt, norm_cols}},
                j <- 0..(num_features - 1) do
            x_j = xt[j]
            norm_j = norm_cols[j]
            w_j = w[j]

            # Residual without the contribution of feature j.
            residual = residual + w_j * x_j
            tmp = Nx.dot(x_j, residual)

            updated = Nx.sign(tmp) * Nx.max(Nx.abs(tmp) - l1_reg, 0) / (norm_j + l2_reg)
            # A zero column carries no information, its coefficient stays at zero.
            new_w_j = Nx.select(norm_j == 0, w_j, updated)

            residual = residual - new_w_j * x_j
            w = Nx.put_slice(w, [j], Nx.reshape(new_w_j, {1}))

            {{w, residual, Nx.max(w_max, Nx.abs(new_w_j)),
              Nx.max(d_w_max, Nx.abs(new_w_j - w_j))}, {xt, norm_cols}}
          end

        check_gap? =
          w_max == 0 or d_w_max / w_max < d_w_tol or iteration == max_iterations - 1

        gap =
          if check_gap? do
            duality_gap(xt, y, w, residual, l1_reg, l2_reg)
          else
            gap
          end

        converged? = check_gap? and gap < gap_tol

        {{w, residual, gap, iteration + 1, converged?}, {xt, y, norm_cols, gap_tol}}
      end

    {w, gap, iteration}
  end

  defnp duality_gap(xt, y, w, residual, l1_reg, l2_reg) do
    xta = Nx.dot(xt, residual) - l2_reg * w
    dual_norm_xta = Nx.reduce_max(Nx.abs(xta))

    r_norm2 = Nx.dot(residual, residual)
    w_norm2 = Nx.dot(w, w)

    scale? = dual_norm_xta > l1_reg
    const = Nx.select(scale?, l1_reg / dual_norm_xta, 1.0)
    gap = Nx.select(scale?, 0.5 * (r_norm2 + r_norm2 * const ** 2), r_norm2)

    gap + l1_reg * Nx.sum(Nx.abs(w)) - const * Nx.dot(residual, y) +
      0.5 * l2_reg * (1 + const ** 2) * w_norm2
  end

  @doc """
  Makes predictions with the given `model` on input `x`.

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
