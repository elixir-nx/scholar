defmodule Scholar.Linear.Lasso do
  @moduledoc ~S"""
  Linear regression with $L_1$ regularization.

  Minimizes the objective function:
  $$
  \frac{1}{2n}||y - Xw||^2\_2 + \alpha||w||\_1
  $$

  Where:
  * $X$ is an input data

  * $y$ is an input target

  * $w$ is the model weights matrix

  * $n$ is the number of samples

  * $\alpha$ is the parameter that controls the level of regularization

  The $L_1$ term drives coefficients to exactly zero, so the fitted model is
  sparse, and larger values of $\alpha$ keep fewer features.

  This is `Scholar.Linear.ElasticNet` with `l1_ratio: 1.0`, fitted with the
  same coordinate descent solver. See its documentation for the stopping rule.

  Time complexity is $O(I \cdot N \cdot K)$, where $I$ is the number of
  iterations, $N$ is the number of samples and $K$ is the number of features.
  """
  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Linear.{ElasticNet, LinearHelpers}

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
      Constant that multiplies the $L_1$ term, controlling regularization strength.
      `:alpha` must be a non-negative float i.e. in [0, inf).

      If `:alpha` is set to 0.0 the objective is the ordinary least squares regression.
      In this case, for numerical reasons, you should use `Scholar.Linear.LinearRegression` instead.
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
  Fits a Lasso model for sample inputs `x` and sample targets `y`.

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

      iex> x = Nx.tensor([[1.0, 2.0, 0.5], [3.0, 2.0, -1.0], [4.0, 7.0, 2.0], [0.0, -1.0, 3.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0, 6.0])
      iex> Scholar.Linear.Lasso.fit(x, y, alpha: 0.3)
      %Scholar.Linear.Lasso{
        coefficients: Nx.tensor(
          [-0.14346746, -0.77286434, 0.0]
        ),
        intercept: Nx.tensor(
          5.2190957
        ),
        dual_gap: Nx.tensor(
          1.8060207e-5
        ),
        num_iterations: Nx.tensor(
          36,
          type: :u32
        )
      }
  """
  deftransform fit(x, y, opts \\ []) do
    {num_samples, _} = Nx.shape(x)
    y = LinearHelpers.validate_y_shape(y, num_samples, __MODULE__)

    opts = NimbleOptions.validate!(opts, @opts_schema)

    type = Nx.Type.merge(to_float_type(x), to_float_type(y))
    sample_weights = validate_weights(opts[:sample_weights], num_samples, type: type)

    fit_n(x, y, sample_weights,
      type: type,
      l1_reg: opts[:alpha] * num_samples,
      l2_reg: 0.0,
      tol: opts[:tol],
      max_iterations: opts[:max_iterations],
      fit_intercept?: opts[:fit_intercept?]
    )
  end

  defnp fit_n(x, y, sample_weights, opts) do
    model = ElasticNet.fit_n(x, y, sample_weights, opts)

    %__MODULE__{
      coefficients: model.coefficients,
      intercept: model.intercept,
      dual_gap: model.dual_gap,
      num_iterations: model.num_iterations
    }
  end

  @doc """
  Makes predictions with the given `model` on input `x`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0, 0.5], [3.0, 2.0, -1.0], [4.0, 7.0, 2.0], [0.0, -1.0, 3.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0, 6.0])
      iex> model = Scholar.Linear.Lasso.fit(x, y, alpha: 0.3)
      iex> Scholar.Linear.Lasso.predict(model, Nx.tensor([[2.0, 1.0, 1.0]]))
      Nx.tensor(
        [4.1592965]
      )
  """
  defn predict(%__MODULE__{coefficients: coeff, intercept: intercept} = _model, x) do
    Nx.dot(x, [-1], coeff, [-1]) + intercept
  end
end
