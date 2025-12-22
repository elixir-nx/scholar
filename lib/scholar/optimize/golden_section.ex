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
    bracket: [
      type: {:custom, __MODULE__, :__bracket__, []},
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

  @opts_schema NimbleOptions.new!(opts)

  @doc false
  def __bracket__(value) do
    case value do
      {a, b} when is_number(a) and is_number(b) and a < b ->
        {:ok, {a, b}}

      {a, b} when is_number(a) and is_number(b) ->
        {:error,
         "expected :bracket to be a tuple {a, b} where a < b, got: #{inspect(value)}"}

      _ ->
        {:error,
         "expected :bracket to be a tuple {a, b} of numbers, got: #{inspect(value)}"}
    end
  end

  @doc """
  Minimizes a scalar function using golden section search.

  ## Arguments

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
      iex> result = Scholar.Optimize.GoldenSection.minimize(fun, bracket: {0.0, 5.0})
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor(3.0), atol: 1.0e-4) |> Nx.to_number()
      1
  """
  defn minimize(fun, opts \\ []) do
    {a, b, tol, maxiter} = transform_opts(opts)
    minimize_n(fun, a, b, tol, maxiter)
  end

  deftransformp transform_opts(opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    {a, b} = opts[:bracket]

    a = Nx.tensor(a, type: :f64)
    b = Nx.tensor(b, type: :f64)
    tol = Nx.tensor(opts[:tol], type: :f64)
    maxiter = Nx.tensor(opts[:maxiter], type: :s64)

    {a, b, tol, maxiter}
  end

  defnp minimize_n(fun, a, b, tol, maxiter) do
    # Golden ratio as tensor
    phi = Nx.tensor(@phi, type: :f64)

    # a < b is guaranteed by option validation
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

    %__MODULE__{
      x: x_opt,
      fun: f_opt,
      converged: converged,
      iterations: final_state.iter,
      fun_evals: final_state.f_evals + 1
    }
  end
end
