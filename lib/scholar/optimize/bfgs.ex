defmodule Scholar.Optimize.BFGS do
  @moduledoc """
  BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm for multivariate function minimization.

  BFGS is a quasi-Newton optimization method that approximates the inverse Hessian
  matrix using gradient information. It is well-suited for smooth, differentiable
  objective functions and typically converges much faster than derivative-free methods
  like Nelder-Mead.

  ## Algorithm

  At each iteration, the algorithm:
  1. Computes the gradient using automatic differentiation
  2. Determines a search direction from the inverse Hessian approximation
  3. Performs a line search to find an acceptable step length
  4. Updates the inverse Hessian approximation using the BFGS formula

  ## Convergence

  The algorithm converges when the gradient norm falls below the specified tolerance.

  ## References

  * Nocedal, J. and Wright, S. J. (2006). "Numerical Optimization"
  * Fletcher, R. (1987). "Practical Methods of Optimization"
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:x, :fun, :converged, :iterations, :fun_evals, :grad_evals]}
  defstruct [:x, :fun, :converged, :iterations, :fun_evals, :grad_evals]

  @type t :: %__MODULE__{
          x: Nx.Tensor.t(),
          fun: Nx.Tensor.t(),
          converged: Nx.Tensor.t(),
          iterations: Nx.Tensor.t(),
          fun_evals: Nx.Tensor.t(),
          grad_evals: Nx.Tensor.t()
        }

  # Line search parameter (Armijo condition)
  @c1 1.0e-4

  opts = [
    gtol: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-5,
      doc: """
      Gradient norm tolerance for convergence. The algorithm stops when
      the infinity norm of the gradient is below this threshold.
      """
    ],
    maxiter: [
      type: :pos_integer,
      default: 500,
      doc: "Maximum number of iterations."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Minimizes a multivariate function using the BFGS algorithm.

  BFGS is a gradient-based method that uses automatic differentiation to compute
  gradients and approximates the inverse Hessian matrix for fast convergence.

  ## Arguments

  * `x0` - Initial guess as a 1-D tensor of shape `{n}`.
  * `fun` - The objective function to minimize. Must be a defn-compatible function
    that takes a 1-D tensor of shape `{n}` and returns a scalar tensor.
  * `opts` - Options (see below).

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Returns

  A `Scholar.Optimize.BFGS` struct with the optimization result:

  * `:x` - The optimal point found (shape `{n}`)
  * `:fun` - The function value at the optimal point
  * `:converged` - Whether the optimization converged (1 if true, 0 if false)
  * `:iterations` - Number of iterations performed
  * `:fun_evals` - Number of function evaluations
  * `:grad_evals` - Number of gradient evaluations

  ## Examples

      iex> # Minimize a simple quadratic: f(x) = x1^2 + x2^2
      iex> fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      iex> x0 = Nx.tensor([1.0, 2.0])
      iex> result = Scholar.Optimize.BFGS.minimize(x0, fun)
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor([0.0, 0.0]), atol: 1.0e-4) |> Nx.to_number()
      1

  For higher precision, use f64 tensors:

      iex> fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      iex> x0 = Nx.tensor([1.0, 2.0], type: :f64)
      iex> result = Scholar.Optimize.BFGS.minimize(x0, fun, gtol: 1.0e-10)
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor([0.0, 0.0]), atol: 1.0e-8) |> Nx.to_number()
      1

  Minimizing the Rosenbrock function:

      iex> # Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1, 1)
      iex> rosenbrock = fn x ->
      ...>   x0 = x[0]
      ...>   x1 = x[1]
      ...>   term1 = Nx.pow(Nx.subtract(1, x0), 2)
      ...>   term2 = Nx.multiply(100, Nx.pow(Nx.subtract(x1, Nx.pow(x0, 2)), 2))
      ...>   Nx.add(term1, term2)
      ...> end
      iex> x0 = Nx.tensor([0.0, 0.0], type: :f64)
      iex> result = Scholar.Optimize.BFGS.minimize(x0, rosenbrock, gtol: 1.0e-8, maxiter: 1000)
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor([1.0, 1.0]), atol: 1.0e-4) |> Nx.to_number()
      1
  """
  defn minimize(x0, fun, opts \\ []) do
    {gtol, maxiter} = transform_opts(opts)
    minimize_n(x0, fun, gtol, maxiter)
  end

  deftransformp transform_opts(opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    {opts[:gtol], opts[:maxiter]}
  end

  defnp minimize_n(x0, fun, gtol, maxiter) do
    x0 = Nx.flatten(x0)
    {n} = Nx.shape(x0)

    # Initial function value and gradient
    {f0, g0} = value_and_grad(x0, fun)

    # Initialize inverse Hessian as identity matrix
    h_inv = Nx.eye(n, type: Nx.type(x0))

    # Initial state
    initial_state = %{
      x: x0,
      f: f0,
      g: g0,
      h_inv: h_inv,
      iter: Nx.u32(0),
      f_evals: Nx.u32(1),
      g_evals: Nx.u32(1)
    }

    # Main optimization loop
    {final_state, _} =
      while {state = initial_state, {gtol, maxiter}},
            not converged?(state, gtol) and state.iter < maxiter do
        new_state = bfgs_step(state, fun)
        {new_state, {gtol, maxiter}}
      end

    # Check convergence
    converged = converged?(final_state, gtol)

    %__MODULE__{
      x: final_state.x,
      fun: final_state.f,
      converged: converged,
      iterations: final_state.iter,
      fun_evals: final_state.f_evals,
      grad_evals: final_state.g_evals
    }
  end

  # Check convergence: infinity norm of gradient < gtol
  defnp converged?(state, gtol) do
    grad_norm = Nx.reduce_max(Nx.abs(state.g))
    Nx.less(grad_norm, gtol)
  end

  # Perform one BFGS iteration
  defnp bfgs_step(state, fun) do
    x = state.x
    f = state.f
    g = state.g
    h_inv = state.h_inv

    # Compute search direction: p = -H_inv * g
    p = Nx.negate(Nx.dot(h_inv, g))

    # Line search to find step length alpha
    {alpha, f_new, g_new, ls_f_evals, ls_g_evals} =
      line_search(x, f, g, p, fun)

    # Compute step and gradient change
    s = Nx.multiply(alpha, p)
    x_new = Nx.add(x, s)
    y = Nx.subtract(g_new, g)

    # Update inverse Hessian using BFGS formula
    h_inv_new = update_inverse_hessian(h_inv, s, y)

    %{
      state
      | x: x_new,
        f: f_new,
        g: g_new,
        h_inv: h_inv_new,
        iter: Nx.add(state.iter, 1),
        f_evals: Nx.add(state.f_evals, ls_f_evals),
        g_evals: Nx.add(state.g_evals, ls_g_evals)
    }
  end

  # Backtracking line search with Armijo condition (unrolled for defn compatibility)
  defnp line_search(x, f, g, p, fun) do
    # Directional derivative at alpha=0
    slope = Nx.dot(g, p)

    # Try step sizes: 1, 0.5, 0.25, 0.125, ...
    # Unroll 10 iterations of backtracking
    alpha = backtrack_step(x, f, p, slope, fun, 1.0)
    alpha = backtrack_step(x, f, p, slope, fun, alpha)
    alpha = backtrack_step(x, f, p, slope, fun, alpha)
    alpha = backtrack_step(x, f, p, slope, fun, alpha)
    alpha = backtrack_step(x, f, p, slope, fun, alpha)
    alpha = backtrack_step(x, f, p, slope, fun, alpha)
    alpha = backtrack_step(x, f, p, slope, fun, alpha)
    alpha = backtrack_step(x, f, p, slope, fun, alpha)
    alpha = backtrack_step(x, f, p, slope, fun, alpha)
    alpha = backtrack_step(x, f, p, slope, fun, alpha)

    # Evaluate function and gradient at final alpha
    x_final = Nx.add(x, Nx.multiply(alpha, p))
    {f_final, g_final} = value_and_grad(x_final, fun)

    {alpha, f_final, g_final, Nx.u32(11), Nx.u32(1)}
  end

  # Single backtracking step
  defnp backtrack_step(x, f, p, slope, fun, alpha) do
    x_trial = Nx.add(x, Nx.multiply(alpha, p))
    f_trial = fun.(x_trial)

    # Check Armijo condition
    armijo_ok =
      Nx.less_equal(f_trial, Nx.add(f, Nx.multiply(@c1, Nx.multiply(alpha, slope))))

    # If satisfied, keep alpha; otherwise halve it
    Nx.select(armijo_ok, alpha, Nx.multiply(0.5, alpha))
  end

  # BFGS inverse Hessian update
  # H_new = (I - rho*s*y') * H * (I - rho*y*s') + rho*s*s'
  defnp update_inverse_hessian(h_inv, s, y) do
    {n} = Nx.shape(s)

    # rho = 1 / (y' * s)
    ys = Nx.dot(y, s)

    # Skip update if ys is too small (would cause numerical issues)
    skip_update = Nx.less(Nx.abs(ys), 1.0e-10)

    rho = Nx.select(skip_update, 0.0, Nx.divide(1.0, ys))

    # Compute update terms
    # I - rho*s*y' and I - rho*y*s'
    eye = Nx.eye(n, type: Nx.type(s))

    # s and y as column vectors for outer products
    s_col = Nx.reshape(s, {n, 1})
    y_col = Nx.reshape(y, {n, 1})
    s_row = Nx.reshape(s, {1, n})
    y_row = Nx.reshape(y, {1, n})

    # (I - rho*s*y')
    left = Nx.subtract(eye, Nx.multiply(rho, Nx.dot(s_col, y_row)))

    # (I - rho*y*s')
    right = Nx.subtract(eye, Nx.multiply(rho, Nx.dot(y_col, s_row)))

    # rho*s*s'
    ss_term = Nx.multiply(rho, Nx.dot(s_col, s_row))

    # H_new = left * H * right + ss_term
    h_new = Nx.add(Nx.dot(Nx.dot(left, h_inv), right), ss_term)

    # Use old H if update skipped
    Nx.select(skip_update, h_inv, h_new)
  end
end
