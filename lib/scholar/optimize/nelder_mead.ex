defmodule Scholar.Optimize.NelderMead do
  @moduledoc """
  Nelder-Mead simplex algorithm for derivative-free multivariate function minimization.

  The Nelder-Mead algorithm (also known as the downhill simplex method) is a
  derivative-free optimization technique for finding the minimum of a function
  of multiple variables. It maintains a simplex of n+1 points in n-dimensional
  space and iteratively updates the simplex by reflecting, expanding, contracting,
  or shrinking based on function values.

  ## Algorithm

  At each iteration, the algorithm:
  1. Orders the simplex vertices by function value (best to worst)
  2. Computes the centroid of all vertices except the worst
  3. Attempts reflection of the worst point through the centroid
  4. Based on the reflected point's value:
     - If best so far: try expansion
     - If better than second-worst: accept reflection
     - If worse: try contraction
     - If contraction fails: shrink simplex toward best point

  ## Convergence

  The algorithm converges when the standard deviation of function values
  at simplex vertices falls below the specified tolerance.

  ## References

  * Nelder, J. A. and Mead, R. (1965). "A simplex method for function minimization"
  * Press, W. H., et al. "Numerical Recipes: The Art of Scientific Computing"
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

  # Standard Nelder-Mead coefficients
  @rho 1.0
  @chi 2.0
  @gamma 0.5
  @sigma 0.5

  opts = [
    tol: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-5,
      doc: """
      Tolerance for convergence. The algorithm stops when the standard deviation
      of function values at simplex vertices is below this threshold.
      """
    ],
    maxiter: [
      type: :pos_integer,
      default: 500,
      doc: "Maximum number of iterations."
    ],
    initial_simplex_step: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 0.05,
      doc: """
      Step size for constructing the initial simplex. Each vertex (except the first)
      is created by moving along one coordinate axis by this factor times the
      corresponding coordinate value (or by this value if the coordinate is zero).
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Minimizes a multivariate function using the Nelder-Mead simplex algorithm.

  This is a derivative-free method suitable for optimizing functions where
  gradients are unavailable or expensive to compute.

  ## Arguments

  * `x0` - Initial guess as a 1-D tensor of shape `{n}`.
  * `fun` - The objective function to minimize. Must be a defn-compatible function
    that takes a 1-D tensor of shape `{n}` and returns a scalar tensor.
  * `opts` - Options (see below).

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Returns

  A `Scholar.Optimize.NelderMead` struct with the optimization result:

  * `:x` - The optimal point found (shape `{n}`)
  * `:fun` - The function value at the optimal point
  * `:converged` - Whether the optimization converged (1 if true, 0 if false)
  * `:iterations` - Number of iterations performed
  * `:fun_evals` - Number of function evaluations

  ## Examples

      iex> # Minimize a simple quadratic: f(x) = x1^2 + x2^2
      iex> fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      iex> x0 = Nx.tensor([1.0, 2.0])
      iex> result = Scholar.Optimize.NelderMead.minimize(x0, fun)
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor([0.0, 0.0]), atol: 1.0e-2) |> Nx.to_number()
      1

  For higher precision, use f64 tensors:

      iex> fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      iex> x0 = Nx.tensor([1.0, 2.0], type: :f64)
      iex> result = Scholar.Optimize.NelderMead.minimize(x0, fun, tol: 1.0e-12)
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor([0.0, 0.0]), atol: 1.0e-4) |> Nx.to_number()
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
      iex> result = Scholar.Optimize.NelderMead.minimize(x0, rosenbrock, tol: 1.0e-8, maxiter: 1000)
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor([1.0, 1.0]), atol: 1.0e-4) |> Nx.to_number()
      1
  """
  defn minimize(x0, fun, opts \\ []) do
    {tol, maxiter, initial_simplex_step} = transform_opts(opts)
    minimize_n(x0, fun, tol, maxiter, initial_simplex_step)
  end

  deftransformp transform_opts(opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    {opts[:tol], opts[:maxiter], opts[:initial_simplex_step]}
  end

  defnp minimize_n(x0, fun, tol, maxiter, initial_simplex_step) do
    x0 = Nx.flatten(x0)
    {n} = Nx.shape(x0)

    # Build initial simplex: n+1 vertices
    simplex = build_initial_simplex(x0, initial_simplex_step)

    # Evaluate function at all simplex vertices
    f_values = evaluate_all_vertices(simplex, fun)

    # Initial state
    initial_state = %{
      simplex: simplex,
      f_values: f_values,
      iter: Nx.u32(0),
      f_evals: Nx.u32(n + 1)
    }

    # Main optimization loop
    {final_state, _} =
      while {state = initial_state, {tol, maxiter}},
            not converged?(state, tol) and state.iter < maxiter do
        new_state = nelder_mead_step(state, fun)
        {new_state, {tol, maxiter}}
      end

    # Get best point
    order = Nx.argsort(final_state.f_values)
    best_idx = order[0]
    x_opt = final_state.simplex[best_idx]
    f_opt = final_state.f_values[best_idx]

    # Check convergence
    converged = converged?(final_state, tol)

    %__MODULE__{
      x: x_opt,
      fun: f_opt,
      converged: converged,
      iterations: final_state.iter,
      fun_evals: final_state.f_evals
    }
  end

  # Build initial simplex with n+1 vertices
  defnp build_initial_simplex(x0, step) do
    {n} = Nx.shape(x0)
    simplex_shape = {n + 1, n}

    # Broadcast x0 to all rows
    base = Nx.broadcast(x0, simplex_shape)

    # Create identity-like steps for vertices 1 to n
    indices = Nx.iota(simplex_shape, axis: 0)
    col_indices = Nx.iota(simplex_shape, axis: 1)

    # For row i (i > 0), add step to column i-1
    should_add = Nx.equal(Nx.subtract(indices, 1), col_indices)

    # Step size: use step * |x0[j]| if x0[j] != 0, else step
    x0_abs = Nx.abs(x0)
    step_sizes = Nx.select(Nx.greater(x0_abs, 1.0e-10), Nx.multiply(step, x0_abs), step)

    # Broadcast step_sizes to simplex shape
    step_matrix = Nx.broadcast(step_sizes, simplex_shape)

    # Add steps where appropriate
    Nx.select(should_add, Nx.add(base, step_matrix), base)
  end

  # Evaluate function at all simplex vertices
  defnp evaluate_all_vertices(simplex, fun) do
    {n_plus_1, _n} = Nx.shape(simplex)

    # Evaluate first vertex to get type
    f0 = fun.(simplex[0])
    init_f = Nx.broadcast(f0, {n_plus_1})

    {f_values, _} =
      while {f_values = init_f, {simplex, i = Nx.u32(0)}}, i < n_plus_1 do
        vertex = simplex[i]
        f_val = fun.(vertex)
        f_values = Nx.indexed_put(f_values, Nx.new_axis(i, 0), f_val)
        {f_values, {simplex, i + 1}}
      end

    f_values
  end

  # Check convergence: std of function values < tol
  defnp converged?(state, tol) do
    f_values = state.f_values
    mean = Nx.mean(f_values)
    variance = Nx.mean(Nx.pow(Nx.subtract(f_values, mean), 2))
    std = Nx.sqrt(variance)
    Nx.less(std, tol)
  end

  # Perform one Nelder-Mead iteration
  defnp nelder_mead_step(state, fun) do
    simplex = state.simplex
    f_values = state.f_values
    {n_plus_1, _n} = Nx.shape(simplex)

    # Sort vertices by function value
    order = Nx.argsort(f_values)
    sorted_simplex = Nx.take(simplex, order, axis: 0)
    sorted_f = Nx.take(f_values, order)

    # Best, second-worst, and worst points
    f_best = sorted_f[0]
    f_second_worst = sorted_f[n_plus_1 - 2]
    x_worst = sorted_simplex[n_plus_1 - 1]
    f_worst = sorted_f[n_plus_1 - 1]

    # Compute centroid of all points except worst
    centroid = Nx.mean(sorted_simplex[0..-2//1], axes: [0])

    # Reflection: x_r = centroid + rho * (centroid - x_worst)
    x_r = Nx.add(centroid, Nx.multiply(@rho, Nx.subtract(centroid, x_worst)))
    f_r = fun.(x_r)

    # Expansion: x_e = centroid + chi * (x_r - centroid)
    x_e = Nx.add(centroid, Nx.multiply(@chi, Nx.subtract(x_r, centroid)))
    f_e = fun.(x_e)

    # Outside contraction: x_oc = centroid + gamma * (x_r - centroid)
    x_oc = Nx.add(centroid, Nx.multiply(@gamma, Nx.subtract(x_r, centroid)))
    f_oc = fun.(x_oc)

    # Inside contraction: x_ic = centroid - gamma * (centroid - x_worst)
    x_ic = Nx.subtract(centroid, Nx.multiply(@gamma, Nx.subtract(centroid, x_worst)))
    f_ic = fun.(x_ic)

    # Decide action based on f_r
    {new_simplex, new_f_values, extra_evals} =
      nelder_mead_update(
        sorted_simplex,
        sorted_f,
        x_r,
        f_r,
        x_e,
        f_e,
        x_oc,
        f_oc,
        x_ic,
        f_ic,
        f_best,
        f_second_worst,
        f_worst,
        fun
      )

    %{
      state
      | simplex: new_simplex,
        f_values: new_f_values,
        iter: Nx.add(state.iter, 1),
        # 4 base evals (f_r, f_e, f_oc, f_ic) + extra for shrink
        f_evals: Nx.add(state.f_evals, Nx.add(4, extra_evals))
    }
  end

  # Update simplex based on reflection result
  defnp nelder_mead_update(
          sorted_simplex,
          sorted_f,
          x_r,
          f_r,
          x_e,
          f_e,
          x_oc,
          f_oc,
          x_ic,
          f_ic,
          f_best,
          f_second_worst,
          f_worst,
          fun
        ) do
    {n_plus_1, n} = Nx.shape(sorted_simplex)

    # Case 1: f_r < f_best -> try expansion
    # Case 2: f_best <= f_r < f_second_worst -> accept reflection
    # Case 3: f_second_worst <= f_r < f_worst -> outside contraction
    # Case 4: f_r >= f_worst -> inside contraction
    # If contraction fails -> shrink

    # Determine which case applies
    case1 = Nx.less(f_r, f_best)
    case2 = Nx.logical_and(Nx.greater_equal(f_r, f_best), Nx.less(f_r, f_second_worst))
    case3 = Nx.logical_and(Nx.greater_equal(f_r, f_second_worst), Nx.less(f_r, f_worst))
    case4 = Nx.greater_equal(f_r, f_worst)

    # Case 1: expansion - use x_e if better than x_r, else use x_r
    use_expansion = Nx.logical_and(case1, Nx.less(f_e, f_r))
    x_case1 = Nx.select(use_expansion, x_e, x_r)
    f_case1 = Nx.select(use_expansion, f_e, f_r)

    # Case 2: accept reflection
    x_case2 = x_r
    f_case2 = f_r

    # Case 3: outside contraction - use if better than x_r
    contraction_success_3 = Nx.less_equal(f_oc, f_r)

    # Case 4: inside contraction - use if better than x_worst
    contraction_success_4 = Nx.less(f_ic, f_worst)

    # Determine if we need to shrink
    need_shrink =
      Nx.logical_or(
        Nx.logical_and(case3, Nx.logical_not(contraction_success_3)),
        Nx.logical_and(case4, Nx.logical_not(contraction_success_4))
      )

    # Select point for non-shrink cases
    x_new_non_shrink =
      Nx.select(
        case1,
        x_case1,
        Nx.select(
          case2,
          x_case2,
          Nx.select(case3, x_oc, x_ic)
        )
      )

    f_new_non_shrink =
      Nx.select(
        case1,
        f_case1,
        Nx.select(
          case2,
          f_case2,
          Nx.select(case3, f_oc, f_ic)
        )
      )

    # Replace worst point (last row) with new point
    # Use indexed_put with proper indices
    worst_idx = Nx.u32(n_plus_1 - 1)
    row_indices = Nx.broadcast(worst_idx, {n})
    col_indices = Nx.iota({n}, type: :u32)
    indices = Nx.stack([row_indices, col_indices], axis: 1)

    new_simplex_replace = Nx.indexed_put(sorted_simplex, indices, x_new_non_shrink)
    new_f_replace = Nx.indexed_put(sorted_f, Nx.new_axis(worst_idx, 0), f_new_non_shrink)

    # Shrink simplex: x_i = x_best + sigma * (x_i - x_best)
    x_best = sorted_simplex[0]

    shrunk_simplex =
      Nx.add(x_best, Nx.multiply(@sigma, Nx.subtract(sorted_simplex, x_best)))

    # Evaluate shrunk simplex (except x_best which stays the same)
    shrunk_f = evaluate_shrunk_vertices(shrunk_simplex, sorted_f, fun)

    # Select between shrink and non-shrink
    final_simplex = Nx.select(need_shrink, shrunk_simplex, new_simplex_replace)
    final_f = Nx.select(need_shrink, shrunk_f, new_f_replace)

    # Extra function evaluations for shrink: n evals (all except best)
    extra_evals = Nx.select(need_shrink, Nx.u32(n), Nx.u32(0))

    {final_simplex, final_f, extra_evals}
  end

  # Evaluate shrunk simplex vertices (skip first which is unchanged)
  defnp evaluate_shrunk_vertices(shrunk_simplex, original_f, fun) do
    {n_plus_1, _n} = Nx.shape(shrunk_simplex)

    {shrunk_f, _} =
      while {shrunk_f = original_f, {shrunk_simplex, i = Nx.u32(1)}}, i < n_plus_1 do
        vertex = shrunk_simplex[i]
        f_val = fun.(vertex)
        shrunk_f = Nx.indexed_put(shrunk_f, Nx.new_axis(i, 0), f_val)
        {shrunk_f, {shrunk_simplex, i + 1}}
      end

    shrunk_f
  end
end
