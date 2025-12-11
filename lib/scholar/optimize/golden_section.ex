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
    tol = Nx.tensor(tol, type: :f64)
    maxiter = Nx.tensor(maxiter, type: :s64)

    minimize_n(fun, a, b, tol, maxiter)
  end

  defn minimize_n(fun, a, b, tol, maxiter) do
    # Golden ratio as tensor
    phi = Nx.tensor(@phi, type: :f64)

    # Ensure a < b using select
    {a, b} = swap_if_needed(a, b)

    # Compute bracket width
    width = b - a

    # Initial interior points: c and d
    c = b - phi * width
    d = a + phi * width
    fc = fun.(c)
    fd = fun.(d)

    # Initial state
    initial_state = %{
      a: a,
      b: b,
      c: c,
      d: d,
      fc: fc,
      fd: fd,
      iter: Nx.tensor(0, type: :s64),
      f_evals: Nx.tensor(2, type: :s64)
    }

    # Main optimization loop
    {final_state, _} =
      while {state = initial_state, {phi, tol, maxiter}},
            state.iter < maxiter and state.b - state.a >= tol do
        # Determine which side to narrow based on function values
        narrow_left = state.fc < state.fd

        # Compute new points for both cases
        # Case 1 (narrow_left): new interval is [a, d], d becomes new c
        new_b_left = state.d
        new_d_left = state.c
        new_fd_left = state.fc
        new_c_left = new_b_left - phi * (new_b_left - state.a)

        # Case 2 (narrow_right): new interval is [c, b], c becomes new d
        new_a_right = state.c
        new_c_right = state.d
        new_fc_right = state.fd
        new_d_right = new_a_right + phi * (state.b - new_a_right)

        # Select based on condition
        new_a = Nx.select(narrow_left, state.a, new_a_right)
        new_b = Nx.select(narrow_left, new_b_left, state.b)
        new_c = Nx.select(narrow_left, new_c_left, new_c_right)
        new_d = Nx.select(narrow_left, new_d_left, new_d_right)

        # Evaluate function at the new point
        # When narrow_left, we need fc at new_c_left
        # When narrow_right, we need fd at new_d_right
        new_fc_left = fun.(new_c_left)
        new_fd_right = fun.(new_d_right)

        new_fc = Nx.select(narrow_left, new_fc_left, new_fc_right)
        new_fd = Nx.select(narrow_left, new_fd_left, new_fd_right)

        new_state = %{
          a: new_a,
          b: new_b,
          c: new_c,
          d: new_d,
          fc: new_fc,
          fd: new_fd,
          iter: state.iter + 1,
          f_evals: state.f_evals + 1
        }

        {new_state, {phi, tol, maxiter}}
      end

    # Solution is the midpoint of the final bracket
    x_opt = (final_state.a + final_state.b) / 2
    f_opt = fun.(x_opt)

    # Check if we converged (bracket width < tol)
    converged = final_state.b - final_state.a < tol

    %Scholar.Optimize{
      x: x_opt,
      fun: f_opt,
      converged: converged,
      iterations: final_state.iter,
      fun_evals: final_state.f_evals + 1,
      grad_evals: Nx.tensor(0, type: :s64)
    }
  end

  defnp swap_if_needed(a, b) do
    should_swap = a > b
    new_a = Nx.select(should_swap, b, a)
    new_b = Nx.select(should_swap, a, b)
    {new_a, new_b}
  end
end
