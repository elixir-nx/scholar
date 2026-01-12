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

  @derive {Nx.Container, containers: [:x, :fun, :converged, :iterations, :fun_evals]}
  defstruct [:x, :fun, :converged, :iterations, :fun_evals]

  @type t :: %__MODULE__{
          x: Nx.Tensor.t(),
          fun: Nx.Tensor.t(),
          converged: Nx.Tensor.t(),
          iterations: Nx.Tensor.t(),
          fun_evals: Nx.Tensor.t()
        }

  # Golden ratio conjugate: (sqrt(5) - 1) / 2
  @phi 0.6180339887498949

  opts = [
    tol: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-5,
      doc: """
      Absolute tolerance for convergence. Default is 1.0e-5 which works with f32 precision.
      For higher precision, use f64 tensors for bounds and a smaller tolerance.
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
  Minimizes a scalar function using golden section search.

  ## Arguments

  * `a` - Lower bound of the search interval (number or scalar tensor).
  * `b` - Upper bound of the search interval (number or scalar tensor). Must satisfy `a < b`.
  * `fun` - The objective function to minimize. Must be a defn-compatible function
    that takes a scalar tensor and returns a scalar tensor.
  * `opts` - Options (see below).

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Returns

  A `Scholar.Optimize.GoldenSection` struct with the optimization result:

  * `:x` - The optimal point found
  * `:fun` - The function value at the optimal point
  * `:converged` - Whether the optimization converged (1 if true, 0 if false)
  * `:iterations` - Number of iterations performed
  * `:fun_evals` - Number of function evaluations

  ## Examples

      iex> fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end
      iex> result = Scholar.Optimize.GoldenSection.minimize(0.0, 5.0, fun)
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor(3.0), atol: 1.0e-3) |> Nx.to_number()
      1

  For higher precision, use f64 tensors:

      iex> fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end
      iex> a = Nx.tensor(0.0, type: :f64)
      iex> b = Nx.tensor(5.0, type: :f64)
      iex> result = Scholar.Optimize.GoldenSection.minimize(a, b, fun, tol: 1.0e-10)
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor(3.0), atol: 1.0e-8) |> Nx.to_number()
      1
  """
  deftransform minimize(a, b, fun, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    minimize_n(a, b, fun, opts[:tol], opts[:maxiter])
  end

  defnp minimize_n(a, b, fun, tol, maxiter) do
    # Compute bracket width
    width = b - a

    # Initial interior points: c and d
    c = b - @phi * width
    d = a + @phi * width
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
      iter: Nx.u32(0),
      f_evals: Nx.u32(2)
    }

    # Main optimization loop
    {final_state, _} =
      while {state = initial_state, {tol, maxiter}},
            state.iter < maxiter and state.b - state.a >= tol do
        # Determine which side to narrow based on function values
        narrow_left = state.fc < state.fd

        # Compute new points for both cases
        # Case 1 (narrow_left): new interval is [a, d], d becomes new c
        new_b_left = state.d
        new_d_left = state.c
        new_fd_left = state.fc
        new_c_left = new_b_left - @phi * (new_b_left - state.a)

        # Case 2 (narrow_right): new interval is [c, b], c becomes new d
        new_a_right = state.c
        new_c_right = state.d
        new_fc_right = state.fd
        new_d_right = new_a_right + @phi * (state.b - new_a_right)

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

        {new_state, {tol, maxiter}}
      end

    # Solution is the midpoint of the final bracket
    x_opt = (final_state.a + final_state.b) / 2
    f_opt = fun.(x_opt)

    # Check if we converged (bracket width < tol)
    converged = final_state.b - final_state.a < tol

    %__MODULE__{
      x: x_opt,
      fun: f_opt,
      converged: converged,
      iterations: final_state.iter,
      fun_evals: final_state.f_evals + 1
    }
  end
end
