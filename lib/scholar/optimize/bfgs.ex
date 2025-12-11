defmodule Scholar.Optimize.BFGS do
  @moduledoc """
  BFGS quasi-Newton method for unconstrained optimization.

  The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is one of the most
  popular quasi-Newton methods for unconstrained optimization. It iteratively
  builds an approximation to the inverse Hessian matrix using gradient
  information, achieving superlinear convergence near the optimum.

  ## Algorithm

  At each iteration $k$, BFGS:
  1. Computes the search direction $p_k = -H_k \\nabla f(x_k)$ where $H_k$
     approximates the inverse Hessian
  2. Performs a line search to find step size $\\alpha_k$ satisfying Wolfe conditions
  3. Updates the position: $x_{k+1} = x_k + \\alpha_k p_k$
  4. Updates the inverse Hessian approximation using the BFGS formula

  ## BFGS Update Formula

  Given $s_k = x_{k+1} - x_k$ and $y_k = \\nabla f(x_{k+1}) - \\nabla f(x_k)$:

  $$H_{k+1} = (I - \\rho_k s_k y_k^T) H_k (I - \\rho_k y_k s_k^T) + \\rho_k s_k s_k^T$$

  where $\\rho_k = \\frac{1}{y_k^T s_k}$.

  ## Gradient Computation

  This implementation uses automatic differentiation via `Nx.Defn.grad` to compute
  gradients, so you only need to provide the objective function.

  ## Convergence

  BFGS achieves superlinear convergence near a local minimum, typically much
  faster than steepest descent. The algorithm converges when the gradient norm
  falls below the specified tolerance.

  ## References

  * Nocedal, J., & Wright, S. J. (2006). "Numerical Optimization"
  * Fletcher, R. (1987). "Practical Methods of Optimization"
  """

  import Nx.Defn

  # Line search parameters (Armijo condition)
  @c1 1.0e-4  # Sufficient decrease parameter

  @doc """
  Minimizes a multivariate function using the BFGS algorithm.

  ## Arguments

  * `fun` - The objective function to minimize. Must be a defn-compatible function
    that takes a 1D tensor of shape `{n}` and returns a scalar tensor.
  * `x0` - Initial guess tensor of shape `{n}`.
  * `opts` - Options including `:tol` and `:maxiter`.

  ## Returns

  A `Scholar.Optimize` struct with the optimization result.

  ## Notes

  Gradients are computed automatically using `Nx.Defn.grad`. The objective
  function must be differentiable.
  """
  deftransform minimize(fun, x0, opts) do
    n = Nx.axis_size(x0, 0)
    tol = opts[:tol]
    maxiter = opts[:maxiter] || 200

    # Create gradient function using automatic differentiation
    grad_fun = Nx.Defn.grad(fun)

    x0 = Nx.as_type(x0, :f64)

    # Initialize
    f0 = Nx.Defn.jit_apply(fun, [x0])
    g0 = Nx.Defn.jit_apply(grad_fun, [x0])

    # Initialize inverse Hessian approximation as identity
    h_inv0 = Nx.eye(n, type: :f64)

    state = %{
      x: x0,
      f: f0,
      g: g0,
      h_inv: h_inv0,
      iter: 0,
      f_evals: 1,
      g_evals: 1,
      n: n
    }

    bfgs_loop(fun, grad_fun, state, tol, maxiter)
  end

  deftransformp bfgs_loop(fun, grad_fun, state, tol, maxiter) do
    %{x: x, f: f, g: g, h_inv: h_inv, iter: iter, f_evals: f_evals, g_evals: g_evals, n: n} = state

    # Check convergence: gradient norm
    g_norm = Nx.to_number(Nx.sqrt(Nx.sum(Nx.pow(g, 2))))
    converged = g_norm < tol

    if converged or iter >= maxiter do
      %Scholar.Optimize{
        x: x,
        fun: f,
        success: converged,
        iterations: Nx.tensor(iter, type: :s64),
        fun_evals: Nx.tensor(f_evals, type: :s64),
        grad_evals: Nx.tensor(g_evals, type: :s64),
        message: build_message(converged, iter, maxiter)
      }
    else
      # Search direction: p = -H * g
      p = Nx.negate(Nx.dot(h_inv, g))

      # Line search with backtracking (Armijo condition)
      {alpha, f_new, ls_f_evals} = line_search(fun, x, f, g, p)

      # Compute step and gradient difference
      s = Nx.multiply(alpha, p)
      x_new = Nx.add(x, s)
      g_new = Nx.Defn.jit_apply(grad_fun, [x_new])
      y = Nx.subtract(g_new, g)

      # BFGS update for inverse Hessian
      h_inv_new = bfgs_update(h_inv, s, y, n)

      new_state = %{
        x: x_new,
        f: f_new,
        g: g_new,
        h_inv: h_inv_new,
        iter: iter + 1,
        f_evals: f_evals + ls_f_evals,
        g_evals: g_evals + 1,
        n: n
      }

      bfgs_loop(fun, grad_fun, new_state, tol, maxiter)
    end
  end

  defp line_search(fun, x, f, g, p) do
    # Backtracking line search with Armijo condition
    alpha = 1.0
    rho = 0.5  # Reduction factor

    # Directional derivative
    slope = Nx.to_number(Nx.dot(g, p))

    # Target decrease for Armijo condition
    target_decrease = @c1 * slope

    f_val = Nx.to_number(f)

    # Evaluate at initial step
    x_new = Nx.add(x, Nx.multiply(alpha, p))
    f_new = Nx.Defn.jit_apply(fun, [x_new])
    f_new_val = Nx.to_number(f_new)
    f_evals = 1

    # Backtracking loop
    {final_alpha, final_f, final_f_evals} =
      do_backtrack(fun, x, f_val, p, target_decrease, rho, alpha, f_new, f_new_val, f_evals)

    {final_alpha, final_f, final_f_evals}
  end

  defp do_backtrack(fun, x, f_val, p, target_decrease, rho, alpha, f_new, f_new_val, f_evals) do
    if f_new_val <= f_val + alpha * target_decrease or f_evals >= 20 do
      {alpha, f_new, f_evals}
    else
      new_alpha = alpha * rho
      x_try = Nx.add(x, Nx.multiply(new_alpha, p))
      new_f = Nx.Defn.jit_apply(fun, [x_try])
      new_f_val = Nx.to_number(new_f)
      do_backtrack(fun, x, f_val, p, target_decrease, rho, new_alpha, new_f, new_f_val, f_evals + 1)
    end
  end

  defp bfgs_update(h_inv, s, y, n) do
    # BFGS inverse Hessian update
    # H_new = (I - rho*s*y') * H * (I - rho*y*s') + rho*s*s'

    # Compute rho = 1 / (y' * s)
    ys = Nx.to_number(Nx.dot(y, s))

    # Skip update if curvature condition is not satisfied
    if ys < 1.0e-10 do
      h_inv
    else
      rho = 1.0 / ys

      identity = Nx.eye(n, type: :f64)

      # Reshape s and y to column vectors for outer products
      s_col = Nx.reshape(s, {n, 1})
      y_col = Nx.reshape(y, {n, 1})

      # V = I - rho * s * y'
      v = Nx.subtract(identity, Nx.multiply(rho, Nx.dot(s_col, Nx.transpose(y_col))))

      # H_new = V * H * V' + rho * s * s'
      h_inv_new = Nx.add(
        Nx.dot(v, Nx.dot(h_inv, Nx.transpose(v))),
        Nx.multiply(rho, Nx.dot(s_col, Nx.transpose(s_col)))
      )

      h_inv_new
    end
  end

  defp build_message(converged, iterations, maxiter) do
    cond do
      converged ->
        "Optimization converged after #{iterations} iterations."

      iterations >= maxiter ->
        "Maximum iterations (#{maxiter}) reached."

      true ->
        "Optimization terminated."
    end
  end
end
