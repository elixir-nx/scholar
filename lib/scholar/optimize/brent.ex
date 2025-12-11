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

  ## Convergence

  Brent's method typically achieves superlinear convergence near the minimum
  due to parabolic interpolation, while maintaining the guaranteed convergence
  of golden section search.

  ## References

  * Brent, R. P. (1973). "Algorithms for Minimization without Derivatives"
  * Press, W. H., et al. "Numerical Recipes: The Art of Scientific Computing"
  """

  import Nx.Defn

  # Golden ratio for golden section steps
  @golden_ratio 0.3819660112501051

  @doc """
  Minimizes a scalar function using Brent's method.

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

    # Initialize: x is the best point so far, w is second best, v is previous w
    x = Nx.add(a, Nx.multiply(@golden_ratio, Nx.subtract(b, a)))
    fx = Nx.Defn.jit_apply(fun, [x])

    # e is the step taken before the previous one
    # d is the previous step
    e = Nx.tensor(0.0, type: :f64)
    d = Nx.tensor(0.0, type: :f64)

    # w and v start same as x
    state = %{
      a: a, b: b, x: x, w: x, v: x,
      fx: fx, fw: fx, fv: fx,
      d: d, e: e,
      iter: 0, f_evals: 1
    }

    brent_loop(fun, state, tol, maxiter)
  end

  deftransformp brent_loop(fun, state, tol, maxiter) do
    %{a: a, b: b, x: x, w: w, v: v, fx: fx, fw: fw, fv: fv, d: d, e: e, iter: iter, f_evals: f_evals} = state

    # Midpoint of current bracket
    xm = Nx.divide(Nx.add(a, b), 2.0)

    # Tolerance values
    x_val = Nx.to_number(x)
    tol1 = tol * abs(x_val) + 1.0e-10
    tol2 = 2.0 * tol1

    # Check for convergence
    xm_val = Nx.to_number(xm)
    a_val = Nx.to_number(a)
    b_val = Nx.to_number(b)
    converged = abs(x_val - xm_val) <= tol2 - 0.5 * (b_val - a_val)

    if converged or iter >= maxiter do
      %Scholar.Optimize{
        x: x,
        fun: fx,
        converged: Nx.tensor(if(converged, do: 1, else: 0), type: :u8),
        iterations: Nx.tensor(iter, type: :s64),
        fun_evals: Nx.tensor(f_evals, type: :s64),
        grad_evals: Nx.tensor(0, type: :s64)
      }
    else
      # Try parabolic interpolation
      {new_d, new_e, use_golden} = parabolic_step(x, w, v, fx, fw, fv, d, e, a, b, tol1)

      # Apply golden section if parabolic step is not acceptable
      {new_d, new_e} =
        if use_golden do
          golden_step(x, a, b)
        else
          {new_d, new_e}
        end

      # Compute new point
      new_d_val = Nx.to_number(new_d)
      u =
        if abs(new_d_val) >= tol1 do
          Nx.add(x, new_d)
        else
          sign_d = if new_d_val >= 0, do: 1.0, else: -1.0
          Nx.add(x, Nx.tensor(tol1 * sign_d, type: :f64))
        end

      fu = Nx.Defn.jit_apply(fun, [u])

      # Update bracket and best points
      {new_a, new_b, new_x, new_w, new_v, new_fx, new_fw, new_fv} =
        update_bracket(u, fu, x, w, v, fx, fw, fv, a, b)

      new_state = %{
        a: new_a, b: new_b, x: new_x, w: new_w, v: new_v,
        fx: new_fx, fw: new_fw, fv: new_fv,
        d: new_d, e: new_e,
        iter: iter + 1, f_evals: f_evals + 1
      }

      brent_loop(fun, new_state, tol, maxiter)
    end
  end

  defp parabolic_step(x, w, v, fx, fw, fv, d, e, a, b, tol1) do
    x_val = Nx.to_number(x)
    w_val = Nx.to_number(w)
    v_val = Nx.to_number(v)
    fx_val = Nx.to_number(fx)
    fw_val = Nx.to_number(fw)
    fv_val = Nx.to_number(fv)
    _d_val = Nx.to_number(d)
    e_val = Nx.to_number(e)
    a_val = Nx.to_number(a)
    b_val = Nx.to_number(b)

    # Try parabolic interpolation only if e is large enough
    use_parabolic = abs(e_val) > tol1

    if not use_parabolic do
      {Nx.tensor(0.0, type: :f64), d, true}
    else
      # Compute parabolic minimum through x, w, v
      r = (x_val - w_val) * (fx_val - fv_val)
      q = (x_val - v_val) * (fx_val - fw_val)
      p = (x_val - v_val) * q - (x_val - w_val) * r
      q = 2.0 * (q - r)

      # Make q positive and adjust p sign accordingly
      {p, q} = if q > 0, do: {-p, q}, else: {p, -q}

      r_old = e_val
      new_e = d

      # Check if parabolic step is acceptable
      step_acceptable =
        abs(p) < abs(0.5 * q * r_old) and
        p > q * (a_val - x_val) and
        p < q * (b_val - x_val)

      if step_acceptable do
        new_d = Nx.tensor(p / q, type: :f64)
        {new_d, new_e, false}
      else
        {Nx.tensor(0.0, type: :f64), d, true}
      end
    end
  end

  defp golden_step(x, a, b) do
    x_val = Nx.to_number(x)
    a_val = Nx.to_number(a)
    b_val = Nx.to_number(b)

    # Take a golden section step into the larger segment
    new_e_val = if x_val < 0.5 * (a_val + b_val), do: b_val - x_val, else: a_val - x_val
    new_d_val = @golden_ratio * new_e_val

    {Nx.tensor(new_d_val, type: :f64), Nx.tensor(new_e_val, type: :f64)}
  end

  defp update_bracket(u, fu, x, w, v, fx, fw, fv, a, b) do
    u_val = Nx.to_number(u)
    fu_val = Nx.to_number(fu)
    x_val = Nx.to_number(x)
    fx_val = Nx.to_number(fx)
    fw_val = Nx.to_number(fw)
    fv_val = Nx.to_number(fv)
    w_val = Nx.to_number(w)
    v_val = Nx.to_number(v)

    cond do
      fu_val <= fx_val ->
        # u is the new best point - update bracket to exclude the side away from u
        # If u >= x, minimum is in [x, b], so a becomes x
        # If u < x, minimum is in [a, x], so b becomes x
        {new_a, new_b} = if u_val >= x_val, do: {x, b}, else: {a, x}
        {new_a, new_b, u, x, w, fu, fx, fw}

      fu_val <= fw_val or w_val == x_val ->
        # u is second best - update bracket to exclude u's side
        # If u < x, we know minimum is in [u, b] (since x is better than u and x > u)
        # If u >= x, we know minimum is in [a, u]
        {new_a, new_b} = if u_val < x_val, do: {u, b}, else: {a, u}
        {new_a, new_b, x, u, w, fx, fu, fw}

      fu_val <= fv_val or v_val == x_val or v_val == w_val ->
        # u is third best
        {new_a, new_b} = if u_val < x_val, do: {u, b}, else: {a, u}
        {new_a, new_b, x, w, u, fx, fw, fu}

      true ->
        # u is worse than all current points, just update bracket
        {new_a, new_b} = if u_val < x_val, do: {u, b}, else: {a, u}
        {new_a, new_b, x, w, v, fx, fw, fv}
    end
  end

end
