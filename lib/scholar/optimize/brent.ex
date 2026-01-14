defmodule Scholar.Optimize.Brent do
  @moduledoc """
  Brent's method for univariate function minimization.

  Brent's method combines the robustness of golden section search with the
  speed of parabolic interpolation. It uses parabolic interpolation when
  it appears to be converging well, but falls back to golden section search
  when the parabola would produce a step outside the bracket or too small
  a reduction.

  ## Algorithm

  At each iteration, the algorithm considers fitting a parabola through
  three points and using its minimum as the next guess. If the parabolic
  step is acceptable (within bounds and making sufficient progress), it is
  used. Otherwise, a golden section step is taken.

  The algorithm tracks six points:
  - `a`, `b`: Current bracket bounds (minimum is between a and b)
  - `x`: Best point found so far (lowest function value)
  - `w`: Second best point
  - `v`: Previous value of w
  - `d`: Most recent step size
  - `e`: Step size from two iterations ago

  ## Convergence

  Brent's method typically achieves superlinear convergence near the minimum
  due to parabolic interpolation, while maintaining the guaranteed convergence
  of golden section search. It is significantly faster than pure golden section
  for smooth functions.

  ## References

  * Brent, R. P. (1973). "Algorithms for Minimization without Derivatives"
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

  # Golden ratio for golden section steps: (3 - sqrt(5)) / 2
  @golden_ratio 0.3819660112501051

  # Small epsilon for numerical stability
  @eps 1.0e-11

  opts = [
    tol: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-5,
      doc: """
      Relative tolerance for convergence. Default is 1.0e-5 which works with f32 precision.
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
  Minimizes a scalar function using Brent's method.

  Brent's method is the recommended algorithm for scalar optimization as it
  combines the reliability of golden section search with faster convergence
  from parabolic interpolation.

  ## Arguments

  * `a` - Lower bound of the search interval (number or scalar tensor).
  * `b` - Upper bound of the search interval (number or scalar tensor). Must satisfy `a < b`.
  * `fun` - The objective function to minimize. Must be a defn-compatible function
    that takes a scalar tensor and returns a scalar tensor.
  * `opts` - Options (see below).

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Returns

  A `Scholar.Optimize.Brent` struct with the optimization result:

  * `:x` - The optimal point found
  * `:fun` - The function value at the optimal point
  * `:converged` - Whether the optimization converged (1 if true, 0 if false)
  * `:iterations` - Number of iterations performed
  * `:fun_evals` - Number of function evaluations

  ## Examples

      iex> fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end
      iex> result = Scholar.Optimize.Brent.minimize(0.0, 5.0, fun)
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor(3.0), atol: 1.0e-4) |> Nx.to_number()
      1

  For higher precision, use f64 tensors:

      iex> fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end
      iex> a = Nx.tensor(0.0, type: :f64)
      iex> b = Nx.tensor(5.0, type: :f64)
      iex> result = Scholar.Optimize.Brent.minimize(a, b, fun, tol: 1.0e-10)
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor(3.0), atol: 1.0e-8) |> Nx.to_number()
      1

  ## Comparison with Golden Section

  Brent's method typically converges in significantly fewer iterations than
  golden section search:

      # For a simple parabola (x-3)^2 on [0, 5]:
      # Brent: ~5-8 function evaluations
      # Golden Section: ~40-45 function evaluations
  """
  deftransform minimize(a, b, fun, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    minimize_n(a, b, fun, opts[:tol], opts[:maxiter])
  end

  defnp minimize_n(a, b, fun, tol, maxiter) do
    # Initialize x at the golden section point
    # x is the best point so far, w is second best, v is previous w
    x = a + @golden_ratio * (b - a)
    fx = fun.(x)

    # Initial state: w = v = x (all start at the same point)
    initial_state = %{
      a: a,
      b: b,
      x: x,
      w: x,
      v: x,
      fx: fx,
      fw: fx,
      fv: fx,
      d: b - a,
      e: b - a,
      iter: Nx.u32(0),
      f_evals: Nx.u32(1)
    }

    # Main optimization loop
    {final_state, _} =
      while {state = initial_state, {tol, maxiter}},
            not converged?(state, tol) and state.iter < maxiter do
        # Compute tolerance
        tol1 = tol * Nx.abs(state.x) + @eps

        # Try parabolic interpolation
        {d, e, use_parabolic} =
          try_parabolic_step(
            state.x,
            state.w,
            state.v,
            state.fx,
            state.fw,
            state.fv,
            state.d,
            state.e,
            state.a,
            state.b,
            tol1
          )

        # If parabolic step not acceptable, use golden section
        {d, e} = golden_section_fallback(state.x, state.a, state.b, d, e, use_parabolic)

        # Compute new point u
        # If |d| >= tol1, use d; otherwise use tol1 with sign of d
        u =
          Nx.select(
            Nx.abs(d) >= tol1,
            state.x + d,
            state.x + Nx.select(d >= 0, tol1, -tol1)
          )

        fu = fun.(u)

        # Update bracket and best points
        new_state = update_state(state, u, fu, d, e)

        {new_state, {tol, maxiter}}
      end

    # Check if we converged
    converged = converged?(final_state, tol)

    %__MODULE__{
      x: final_state.x,
      fun: final_state.fx,
      converged: converged,
      iterations: final_state.iter,
      fun_evals: final_state.f_evals
    }
  end

  # Check convergence: |x - midpoint| <= 2*tol1 - (b-a)/2
  defnp converged?(state, tol) do
    xm = (state.a + state.b) / 2
    tol1 = tol * Nx.abs(state.x) + @eps
    tol2 = 2 * tol1
    Nx.abs(state.x - xm) <= tol2 - (state.b - state.a) / 2
  end

  # Try parabolic interpolation
  # Returns {new_d, new_e, use_parabolic}
  defnp try_parabolic_step(x, w, v, fx, fw, fv, d, e, a, b, tol1) do
    # Only try parabolic if e is large enough
    e_large_enough = Nx.abs(e) > tol1

    # Compute parabolic interpolation through x, w, v
    # p/q gives the step from x to the parabola minimum
    r = (x - w) * (fx - fv)
    q = (x - v) * (fx - fw)
    p = (x - v) * q - (x - w) * r
    q = 2 * (q - r)

    # Make q positive, adjust p accordingly
    p = Nx.select(q > 0, -p, p)
    q = Nx.abs(q)

    # Save old e
    e_old = e

    # Check if parabolic step is acceptable:
    # 1. p must be within bounds: q*(a-x) < p < q*(b-x)
    # 2. Step must be smaller than half the step before last: |p| < |0.5*q*e_old|
    in_bounds = p > q * (a - x) and p < q * (b - x)
    small_enough = Nx.abs(p) < Nx.abs(0.5 * q * e_old)

    use_parabolic = e_large_enough and in_bounds and small_enough

    # If using parabolic, d = p/q and e = d (old d)
    # If not using parabolic, we'll compute golden section in the fallback
    new_d = Nx.select(use_parabolic, p / q, d)
    new_e = Nx.select(use_parabolic, d, e)

    {new_d, new_e, use_parabolic}
  end

  # Golden section fallback when parabolic step is not acceptable
  defnp golden_section_fallback(x, a, b, d, e, use_parabolic) do
    # Take golden section step into the larger segment
    # If x < midpoint, step into [x, b]; otherwise into [a, x]
    xm = (a + b) / 2
    new_e = Nx.select(x < xm, b - x, a - x)
    new_d = @golden_ratio * new_e

    # Only use golden section if parabolic was rejected
    final_d = Nx.select(use_parabolic, d, new_d)
    final_e = Nx.select(use_parabolic, e, new_e)

    {final_d, final_e}
  end

  # Update state after evaluating function at u
  defnp update_state(state, u, fu, d, e) do
    # Update bracket: if u < x, new bracket is [a, x] or [x, b]
    u_less_than_x = u < state.x

    # If fu <= fx, u becomes the new best point
    # The old x becomes w, and we update the bracket
    fu_le_fx = fu <= state.fx

    # If fu > fx but fu <= fw (or w == x), u becomes second best
    fu_le_fw = fu <= state.fw
    w_eq_x = state.w == state.x

    # If fu > fw but fu <= fv (or v == x or v == w), u becomes third
    fu_le_fv = fu <= state.fv
    v_eq_x = state.v == state.x
    v_eq_w = state.v == state.w

    # Case 1: fu <= fx - u is new best
    # Update bracket based on which side u is on
    new_a_case1 = Nx.select(u_less_than_x, state.a, state.x)
    new_b_case1 = Nx.select(u_less_than_x, state.x, state.b)

    # Case 2-4: fu > fx - x stays best, just update bracket
    new_a_case234 = Nx.select(u_less_than_x, u, state.a)
    new_b_case234 = Nx.select(u_less_than_x, state.b, u)

    # Select based on fu <= fx
    new_a = Nx.select(fu_le_fx, new_a_case1, new_a_case234)
    new_b = Nx.select(fu_le_fx, new_b_case1, new_b_case234)

    # Update x, w, v and their function values
    # Case 1: fu <= fx
    new_x_1 = u
    new_fx_1 = fu
    new_w_1 = state.x
    new_fw_1 = state.fx
    new_v_1 = state.w
    new_fv_1 = state.fw

    # Case 2: fu > fx but (fu <= fw or w == x)
    case2_condition = fu_le_fw or w_eq_x
    new_x_2 = state.x
    new_fx_2 = state.fx
    new_w_2 = u
    new_fw_2 = fu
    new_v_2 = state.w
    new_fv_2 = state.fw

    # Case 3: fu > fw but (fu <= fv or v == x or v == w)
    case3_condition = fu_le_fv or v_eq_x or v_eq_w
    new_x_3 = state.x
    new_fx_3 = state.fx
    new_w_3 = state.w
    new_fw_3 = state.fw
    new_v_3 = u
    new_fv_3 = fu

    # Case 4: fu > fv - don't update w, v
    new_x_4 = state.x
    new_fx_4 = state.fx
    new_w_4 = state.w
    new_fw_4 = state.fw
    new_v_4 = state.v
    new_fv_4 = state.fv

    # Combine cases: Case 1 > Case 2 > Case 3 > Case 4
    # Start from case 4 (default) and override
    {new_x, new_fx, new_w, new_fw, new_v, new_fv} =
      select_update_case(
        fu_le_fx,
        case2_condition,
        case3_condition,
        {new_x_1, new_fx_1, new_w_1, new_fw_1, new_v_1, new_fv_1},
        {new_x_2, new_fx_2, new_w_2, new_fw_2, new_v_2, new_fv_2},
        {new_x_3, new_fx_3, new_w_3, new_fw_3, new_v_3, new_fv_3},
        {new_x_4, new_fx_4, new_w_4, new_fw_4, new_v_4, new_fv_4}
      )

    %{
      state
      | a: new_a,
        b: new_b,
        x: new_x,
        w: new_w,
        v: new_v,
        fx: new_fx,
        fw: new_fw,
        fv: new_fv,
        d: d,
        e: e,
        iter: state.iter + 1,
        f_evals: state.f_evals + 1
    }
  end

  # Select which case to use for updating x, w, v
  defnp select_update_case(
          fu_le_fx,
          case2_cond,
          case3_cond,
          {x1, fx1, w1, fw1, v1, fv1},
          {x2, fx2, w2, fw2, v2, fv2},
          {x3, fx3, w3, fw3, v3, fv3},
          {x4, fx4, w4, fw4, v4, fv4}
        ) do
    # Priority: Case 1 > Case 2 > Case 3 > Case 4
    # fu_le_fx -> Case 1
    # not fu_le_fx and case2_cond -> Case 2
    # not fu_le_fx and not case2_cond and case3_cond -> Case 3
    # otherwise -> Case 4

    # Select between Case 3 and Case 4
    x_34 = Nx.select(case3_cond, x3, x4)
    fx_34 = Nx.select(case3_cond, fx3, fx4)
    w_34 = Nx.select(case3_cond, w3, w4)
    fw_34 = Nx.select(case3_cond, fw3, fw4)
    v_34 = Nx.select(case3_cond, v3, v4)
    fv_34 = Nx.select(case3_cond, fv3, fv4)

    # Select between Case 2 and (Case 3/4)
    x_234 = Nx.select(case2_cond, x2, x_34)
    fx_234 = Nx.select(case2_cond, fx2, fx_34)
    w_234 = Nx.select(case2_cond, w2, w_34)
    fw_234 = Nx.select(case2_cond, fw2, fw_34)
    v_234 = Nx.select(case2_cond, v2, v_34)
    fv_234 = Nx.select(case2_cond, fv2, fv_34)

    # Select between Case 1 and (Case 2/3/4)
    x_final = Nx.select(fu_le_fx, x1, x_234)
    fx_final = Nx.select(fu_le_fx, fx1, fx_234)
    w_final = Nx.select(fu_le_fx, w1, w_234)
    fw_final = Nx.select(fu_le_fx, fw1, fw_234)
    v_final = Nx.select(fu_le_fx, v1, v_234)
    fv_final = Nx.select(fu_le_fx, fv1, fv_234)

    {x_final, fx_final, w_final, fw_final, v_final, fv_final}
  end
end
