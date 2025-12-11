defmodule Scholar.Optimize.NelderMead do
  @moduledoc """
  Nelder-Mead simplex algorithm for derivative-free optimization.

  The Nelder-Mead method (also known as the downhill simplex method) is a
  direct search method that does not require gradient information. It maintains
  a simplex of $n+1$ points in $n$-dimensional space and iteratively modifies
  it through reflection, expansion, contraction, and shrinkage operations.

  ## Algorithm

  At each iteration, the algorithm:
  1. Orders vertices by function value: $f(x_1) \\leq f(x_2) \\leq ... \\leq f(x_{n+1})$
  2. Computes the centroid of all points except the worst
  3. Attempts reflection of the worst point through the centroid
  4. Based on the reflected point's value, performs expansion, contraction, or accepts reflection
  5. If all else fails, shrinks the simplex toward the best point

  ## Parameters

  The standard coefficients are:
  * Reflection ($\\rho$): 1.0
  * Expansion ($\\chi$): 2.0
  * Contraction ($\\gamma$): 0.5
  * Shrinkage ($\\sigma$): 0.5

  ## Convergence

  The algorithm converges when the simplex becomes sufficiently small, typically
  measured by the standard deviation of function values at the vertices or the
  diameter of the simplex.

  ## References

  * Nelder, J. A., & Mead, R. (1965). "A Simplex Method for Function Minimization"
  * Lagarias, J. C., et al. (1998). "Convergence Properties of the Nelder-Mead Simplex Method"
  """

  import Nx.Defn

  # Standard Nelder-Mead coefficients
  @rho 1.0   # Reflection coefficient
  @chi 2.0   # Expansion coefficient
  @gamma 0.5 # Contraction coefficient
  @sigma 0.5 # Shrinkage coefficient

  @doc """
  Minimizes a multivariate function using the Nelder-Mead simplex method.

  ## Arguments

  * `fun` - The objective function to minimize. Must take a 1D tensor of shape `{n}`
    and return a scalar tensor.
  * `x0` - Initial guess tensor of shape `{n}`.
  * `opts` - Options including `:tol` and `:maxiter`.

  ## Returns

  A `Scholar.Optimize` struct with the optimization result.
  """
  deftransform minimize(fun, x0, opts) do
    n = Nx.axis_size(x0, 0)
    tol = opts[:tol]
    maxiter = opts[:maxiter] || 200 * n

    x0 = Nx.as_type(x0, :f64)

    # Initialize simplex and evaluate at all vertices
    simplex = initialize_simplex(x0, n)
    f_simplex = evaluate_all_vertices(fun, simplex, n)

    # Sort simplex by function values
    {simplex, f_simplex} = sort_simplex(simplex, f_simplex)

    state = %{
      simplex: simplex,
      f_simplex: f_simplex,
      iter: 0,
      f_evals: n + 1,
      n: n
    }

    nelder_mead_loop(fun, state, tol, maxiter)
  end

  defp initialize_simplex(x0, n) do
    # Adaptive step size based on x0
    x0_list = Nx.to_flat_list(x0)

    step =
      Enum.map(x0_list, fn val ->
        if abs(val) > 1.0e-4, do: 0.05 * val, else: 0.00025
      end)

    # Create n+1 vertices: first is x0, others are x0 + step along each axis
    vertices =
      for i <- 0..n do
        if i == 0 do
          x0
        else
          x0_list_updated =
            x0_list
            |> Enum.with_index()
            |> Enum.map(fn {val, idx} ->
              if idx == i - 1, do: val + Enum.at(step, idx), else: val
            end)

          Nx.tensor(x0_list_updated, type: :f64)
        end
      end

    Nx.stack(vertices)
  end

  defp evaluate_all_vertices(fun, simplex, n) do
    values =
      for i <- 0..n do
        vertex = simplex[i]
        Nx.Defn.jit_apply(fun, [vertex])
      end

    Nx.stack(values)
  end

  defp sort_simplex(simplex, f_simplex) do
    indices = Nx.argsort(f_simplex)
    sorted_simplex = Nx.take(simplex, indices, axis: 0)
    sorted_f = Nx.take(f_simplex, indices)
    {sorted_simplex, sorted_f}
  end

  deftransformp nelder_mead_loop(fun, state, tol, maxiter) do
    %{simplex: simplex, f_simplex: f_simplex, iter: iter, f_evals: f_evals, n: n} = state

    # Check convergence: standard deviation of function values
    f_std = Nx.to_number(Nx.standard_deviation(f_simplex))
    converged = f_std < tol

    if converged or iter >= maxiter do
      x_opt = simplex[0]
      f_opt = f_simplex[0]

      %Scholar.Optimize{
        x: x_opt,
        fun: f_opt,
        converged: Nx.tensor(if(converged, do: 1, else: 0), type: :u8),
        iterations: Nx.tensor(iter, type: :s64),
        fun_evals: Nx.tensor(f_evals, type: :s64),
        grad_evals: Nx.tensor(0, type: :s64)
      }
    else
      # Compute centroid of all points except the worst (last)
      centroid = Nx.mean(simplex[0..(n - 1)//1], axes: [0])

      # Get best, second worst, and worst values
      f_best = Nx.to_number(f_simplex[0])
      f_second_worst = Nx.to_number(f_simplex[n - 1])
      f_worst = Nx.to_number(f_simplex[n])

      # Reflection: x_r = centroid + rho * (centroid - x_worst)
      x_worst = simplex[n]
      x_r = Nx.add(centroid, Nx.multiply(@rho, Nx.subtract(centroid, x_worst)))
      f_r = Nx.to_number(Nx.Defn.jit_apply(fun, [x_r]))

      {new_simplex, new_f_simplex, new_f_evals} =
        cond do
          f_r < f_best ->
            # Try expansion
            x_e = Nx.add(centroid, Nx.multiply(@chi, Nx.subtract(x_r, centroid)))
            f_e = Nx.to_number(Nx.Defn.jit_apply(fun, [x_e]))

            if f_e < f_r do
              # Accept expansion
              {update_worst(simplex, x_e, n), update_worst_f(f_simplex, f_e, n), f_evals + 2}
            else
              # Accept reflection
              {update_worst(simplex, x_r, n), update_worst_f(f_simplex, f_r, n), f_evals + 2}
            end

          f_r < f_second_worst ->
            # Accept reflection
            {update_worst(simplex, x_r, n), update_worst_f(f_simplex, f_r, n), f_evals + 1}

          f_r < f_worst ->
            # Outside contraction
            x_c = Nx.add(centroid, Nx.multiply(@gamma, Nx.subtract(x_r, centroid)))
            f_c = Nx.to_number(Nx.Defn.jit_apply(fun, [x_c]))

            if f_c <= f_r do
              {update_worst(simplex, x_c, n), update_worst_f(f_simplex, f_c, n), f_evals + 2}
            else
              # Shrink
              {new_simplex, new_f_simplex} = shrink_simplex(fun, simplex, n)
              {new_simplex, new_f_simplex, f_evals + 2 + n}
            end

          true ->
            # Inside contraction
            x_c = Nx.subtract(centroid, Nx.multiply(@gamma, Nx.subtract(centroid, x_worst)))
            f_c = Nx.to_number(Nx.Defn.jit_apply(fun, [x_c]))

            if f_c < f_worst do
              {update_worst(simplex, x_c, n), update_worst_f(f_simplex, f_c, n), f_evals + 1}
            else
              # Shrink
              {new_simplex, new_f_simplex} = shrink_simplex(fun, simplex, n)
              {new_simplex, new_f_simplex, f_evals + 1 + n}
            end
        end

      # Sort simplex by function values
      {sorted_simplex, sorted_f} = sort_simplex(new_simplex, new_f_simplex)

      new_state = %{
        simplex: sorted_simplex,
        f_simplex: sorted_f,
        iter: iter + 1,
        f_evals: new_f_evals,
        n: n
      }

      nelder_mead_loop(fun, new_state, tol, maxiter)
    end
  end

  defp update_worst(simplex, new_point, n) do
    # Replace the worst point (last) with new_point
    Nx.put_slice(simplex, [n, 0], Nx.reshape(new_point, {1, :auto}))
  end

  defp update_worst_f(f_simplex, new_f, n) do
    # Replace the worst function value (last) with new_f
    f_list = Nx.to_flat_list(f_simplex)
    new_list = List.replace_at(f_list, n, new_f)
    Nx.tensor(new_list, type: Nx.type(f_simplex))
  end

  defp shrink_simplex(fun, simplex, n) do
    # Shrink all points toward the best point
    x_best = simplex[0]

    # Build new simplex by shrinking all points toward best
    {new_vertices, new_values} =
      Enum.map(0..n, fn i ->
        point =
          if i == 0 do
            x_best
          else
            Nx.add(x_best, Nx.multiply(@sigma, Nx.subtract(simplex[i], x_best)))
          end

        f_val = Nx.Defn.jit_apply(fun, [point])
        {point, f_val}
      end)
      |> Enum.unzip()

    new_simplex = Nx.stack(new_vertices)
    new_f_simplex = Nx.stack(new_values)

    {new_simplex, new_f_simplex}
  end

end
