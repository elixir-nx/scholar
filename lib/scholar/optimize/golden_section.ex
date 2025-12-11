defmodule Scholar.Optimize.GoldenSection do
  @moduledoc """
  Golden section search for univariate function minimization.

  Golden section search is a derivative-free optimization technique for
  finding the minimum of a unimodal function within a specified interval.
  It works by iteratively narrowing the bracket using the golden ratio
  to determine probe points.

  ## Algorithm

  The golden ratio $\\phi = \\frac{\\sqrt{5} - 1}{2} \\approx 0.618$ is used to
  select interior points that maintain optimal bracket reduction per iteration.
  At each step, one of the interior points is reused, requiring only one new
  function evaluation per iteration.

  ## Convergence

  The bracket width decreases by a factor of $\\phi \\approx 0.618$ per iteration,
  giving linear convergence. After $n$ iterations, the bracket width is
  approximately $\\phi^n$ times the original width.

  ## References

  * Press, W. H., et al. "Numerical Recipes: The Art of Scientific Computing"
  * Kiefer, J. (1953). "Sequential minimax search for a maximum"
  """

  import Nx.Defn

  # Golden ratio conjugate: (sqrt(5) - 1) / 2
  @phi 0.6180339887498949

  @doc """
  Minimizes a scalar function using golden section search.

  ## Arguments

  * `fun` - The objective function to minimize. Must take a scalar tensor
    and return a scalar tensor.
  * `opts` - Options including `:bracket`, `:tol`, and `:maxiter`.

  ## Returns

  A `Scholar.Optimize` struct with the optimization result.
  """
  deftransform minimize(fun, opts) do
    {a, b} = opts[:bracket]
    tol = opts[:tol]
    maxiter = opts[:maxiter]

    a = Nx.tensor(a, type: :f64)
    b = Nx.tensor(b, type: :f64)

    # Ensure a < b
    {a, b} = if Nx.to_number(a) > Nx.to_number(b), do: {b, a}, else: {a, b}

    # Initial interior points
    c = Nx.subtract(b, Nx.multiply(@phi, Nx.subtract(b, a)))
    d = Nx.add(a, Nx.multiply(@phi, Nx.subtract(b, a)))
    fc = Nx.Defn.jit_apply(fun, [c])
    fd = Nx.Defn.jit_apply(fun, [d])

    # Run the optimization loop
    golden_loop(fun, a, b, c, d, fc, fd, tol, maxiter, 0, 2)
  end

  deftransformp golden_loop(fun, a, b, c, d, fc, fd, tol, maxiter, iter, f_evals) do
    # Check convergence: bracket width
    width = Nx.to_number(Nx.subtract(b, a))
    converged = width < tol

    if converged or iter >= maxiter do
      # Solution is the midpoint of the final bracket
      x_opt = Nx.divide(Nx.add(a, b), 2.0)
      f_opt = Nx.Defn.jit_apply(fun, [x_opt])

      %Scholar.Optimize{
        x: x_opt,
        fun: f_opt,
        success: converged,
        iterations: Nx.tensor(iter, type: :s64),
        fun_evals: Nx.tensor(f_evals + 1, type: :s64),
        grad_evals: Nx.tensor(0, type: :s64),
        message: build_message(converged, iter, maxiter)
      }
    else
      # Narrow the bracket based on function values
      fc_val = Nx.to_number(fc)
      fd_val = Nx.to_number(fd)

      {new_a, new_b, new_c, new_d, new_fc, new_fd} =
        if fc_val < fd_val do
          # Minimum is in [a, d]
          new_b = d
          new_d = c
          new_fd = fc
          new_c = Nx.subtract(new_b, Nx.multiply(@phi, Nx.subtract(new_b, a)))
          new_fc = Nx.Defn.jit_apply(fun, [new_c])
          {a, new_b, new_c, new_d, new_fc, new_fd}
        else
          # Minimum is in [c, b]
          new_a = c
          new_c = d
          new_fc = fd
          new_d = Nx.add(new_a, Nx.multiply(@phi, Nx.subtract(b, new_a)))
          new_fd = Nx.Defn.jit_apply(fun, [new_d])
          {new_a, b, new_c, new_d, new_fc, new_fd}
        end

      golden_loop(fun, new_a, new_b, new_c, new_d, new_fc, new_fd, tol, maxiter, iter + 1, f_evals + 1)
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
