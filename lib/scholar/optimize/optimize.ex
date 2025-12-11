defmodule Scholar.Optimize do
  @moduledoc """
  Optimization routines for minimizing scalar functions.

  This module provides general-purpose optimization functionality similar to
  SciPy's `scipy.optimize`. It supports both derivative-free and gradient-based
  methods.

  ## Multivariate Optimization

  Use `minimize/3` for functions of multiple variables:

      result = Scholar.Optimize.minimize(
        fn x -> Nx.sum(x ** 2) end,
        Nx.tensor([1.0, 2.0, 3.0]),
        method: :nelder_mead
      )

  ## Scalar (Univariate) Optimization

  Use `minimize_scalar/2` for functions of a single variable:

      result = Scholar.Optimize.minimize_scalar(
        fn x -> (x - 2) ** 2 end,
        bracket: {0.0, 4.0}
      )

  ## Available Methods

  ### Multivariate
  * `:nelder_mead` - Derivative-free simplex method (default)
  * `:bfgs` - Quasi-Newton method using automatic differentiation

  ### Scalar
  * `:brent` - Brent's method combining golden section and parabolic interpolation (default)
  * `:golden` - Golden section search
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:x, :fun, :iterations, :fun_evals, :grad_evals]}
  defstruct [:x, :fun, :success, :iterations, :fun_evals, :grad_evals, :message]

  @type t :: %__MODULE__{
          x: Nx.Tensor.t(),
          fun: Nx.Tensor.t(),
          success: boolean(),
          iterations: Nx.Tensor.t(),
          fun_evals: Nx.Tensor.t(),
          grad_evals: Nx.Tensor.t() | nil,
          message: String.t()
        }

  minimize_opts = [
    method: [
      type: {:in, [:nelder_mead, :bfgs]},
      default: :nelder_mead,
      doc: """
      Optimization method to use:
      * `:nelder_mead` - Derivative-free simplex method (default)
      * `:bfgs` - Quasi-Newton method using automatic differentiation
      """
    ],
    tol: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-8,
      doc: "Absolute tolerance for convergence."
    ],
    maxiter: [
      type: {:or, [:pos_integer, nil]},
      default: nil,
      doc: "Maximum number of iterations. If nil, uses method-specific default."
    ],
    learning_loop_unroll: [
      type: :boolean,
      default: false,
      doc: "If true, the optimization loop is unrolled for potentially better JIT performance."
    ]
  ]

  minimize_scalar_opts = [
    method: [
      type: {:in, [:brent, :golden]},
      default: :brent,
      doc: """
      Optimization method to use:
      * `:brent` - Brent's method (default)
      * `:golden` - Golden section search
      """
    ],
    bracket: [
      type: {:custom, Scholar.Options, :bracket, []},
      required: true,
      doc: """
      Bracket containing the minimum. A tuple `{a, b}` defining the search interval.
      The minimum must lie within this interval.
      """
    ],
    tol: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-8,
      doc: "Absolute tolerance for convergence."
    ],
    maxiter: [
      type: :pos_integer,
      default: 500,
      doc: "Maximum number of iterations."
    ]
  ]

  @minimize_schema NimbleOptions.new!(minimize_opts)
  @minimize_scalar_schema NimbleOptions.new!(minimize_scalar_opts)

  @doc """
  Minimize a multivariate scalar function.

  ## Arguments

  * `fun` - The objective function to minimize. Must be a defn-compatible function
    that takes a 1D tensor and returns a scalar tensor.
  * `x0` - Initial guess tensor of shape `{n}` where n is the number of variables.
  * `opts` - Options (see below).

  ## Options

  #{NimbleOptions.docs(@minimize_schema)}

  ## Returns

  A `Scholar.Optimize` struct containing:
  * `:x` - The solution tensor
  * `:fun` - The function value at the solution
  * `:success` - Whether optimization converged successfully
  * `:iterations` - Number of iterations performed
  * `:fun_evals` - Number of function evaluations
  * `:grad_evals` - Number of gradient evaluations (for gradient-based methods)
  * `:message` - Description of the termination reason

  ## Examples

      iex> fun = fn x -> Nx.sum(x ** 2) end
      iex> x0 = Nx.tensor([1.0, 2.0])
      iex> result = Scholar.Optimize.minimize(fun, x0)
      iex> result.success
      true
      iex> Nx.all_close(result.x, Nx.tensor([0.0, 0.0]), atol: 1.0e-4) |> Nx.to_number()
      1
  """
  deftransform minimize(fun, x0, opts \\ []) do
    if Nx.rank(x0) != 1 do
      raise ArgumentError,
            "expected x0 to have shape {n}, got tensor with shape: #{inspect(Nx.shape(x0))}"
    end

    opts = NimbleOptions.validate!(opts, @minimize_schema)

    case opts[:method] do
      :nelder_mead ->
        Scholar.Optimize.NelderMead.minimize(fun, x0, opts)

      :bfgs ->
        Scholar.Optimize.BFGS.minimize(fun, x0, opts)
    end
  end

  @doc """
  Minimize a scalar function of one variable.

  ## Arguments

  * `fun` - The objective function to minimize. Must be a defn-compatible function
    that takes a scalar tensor and returns a scalar tensor.
  * `opts` - Options (see below).

  ## Options

  #{NimbleOptions.docs(@minimize_scalar_schema)}

  ## Returns

  A `Scholar.Optimize` struct where `:x` is a scalar tensor.

  ## Examples

      iex> fun = fn x -> (x - 3) ** 2 end
      iex> result = Scholar.Optimize.minimize_scalar(fun, bracket: {0.0, 5.0})
      iex> result.success
      true
      iex> Nx.all_close(result.x, Nx.tensor(3.0), atol: 1.0e-4) |> Nx.to_number()
      1
  """
  deftransform minimize_scalar(fun, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @minimize_scalar_schema)

    case opts[:method] do
      :golden ->
        Scholar.Optimize.GoldenSection.minimize(fun, opts)

      :brent ->
        Scholar.Optimize.Brent.minimize(fun, opts)
    end
  end
end
