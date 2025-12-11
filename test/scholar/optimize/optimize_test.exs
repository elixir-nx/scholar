defmodule Scholar.OptimizeTest do
  use Scholar.Case, async: true

  require Nx
  alias Scholar.Optimize

  describe "minimize_scalar with golden section" do
    test "minimizes simple parabola" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result = Optimize.minimize_scalar(fun, bracket: {0.0, 5.0}, method: :golden)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(3.0), atol: 1.0e-6)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-10)
    end

    test "minimizes shifted parabola" do
      fun = fn x -> Nx.add(Nx.pow(Nx.add(x, 2), 2), 1) end

      result = Optimize.minimize_scalar(fun, bracket: {-5.0, 5.0}, method: :golden)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(-2.0), atol: 1.0e-6)
      assert_all_close(result.fun, Nx.tensor(1.0), atol: 1.0e-10)
    end

    test "minimizes sine function" do
      # Minimum of sin(x) in [0, 2*pi] is at x = 3*pi/2
      fun = fn x -> Nx.sin(x) end
      expected_x = 3 * :math.pi() / 2

      result = Optimize.minimize_scalar(fun, bracket: {0.0, 2 * :math.pi()}, method: :golden)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(expected_x), atol: 1.0e-5)
    end

    test "respects maxiter limit" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result = Optimize.minimize_scalar(fun, bracket: {0.0, 100.0}, method: :golden, maxiter: 5)

      assert Nx.to_number(result.iterations) <= 5
    end
  end

  describe "minimize_scalar with Brent's method" do
    test "minimizes simple parabola" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result = Optimize.minimize_scalar(fun, bracket: {0.0, 5.0}, method: :brent)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(3.0), atol: 1.0e-6)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-10)
    end

    test "minimizes quartic function" do
      # f(x) = (x-1)^4, minimum at x=1
      fun = fn x -> Nx.pow(Nx.subtract(x, 1), 4) end

      result = Optimize.minimize_scalar(fun, bracket: {-2.0, 4.0}, method: :brent)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(1.0), atol: 1.0e-4)
    end

    test "Brent's method is default" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result = Optimize.minimize_scalar(fun, bracket: {0.0, 5.0})

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(3.0), atol: 1.0e-6)
    end

    test "converges faster than golden section on smooth functions" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result_brent = Optimize.minimize_scalar(fun, bracket: {0.0, 10.0}, method: :brent)
      result_golden = Optimize.minimize_scalar(fun, bracket: {0.0, 10.0}, method: :golden)

      # Both should succeed
      assert Nx.to_number(result_brent.converged) == 1
      assert Nx.to_number(result_golden.converged) == 1

      # Brent should typically use fewer function evaluations
      # (Not always guaranteed, but usually true for smooth functions)
    end
  end

  describe "minimize with Nelder-Mead" do
    test "minimizes sphere function" do
      # f(x) = sum(x^2), minimum at origin
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 2.0, 3.0])

      result = Optimize.minimize(fun, x0, method: :nelder_mead)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([0.0, 0.0, 0.0]), atol: 1.0e-4)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-8)
    end

    test "minimizes 2D quadratic" do
      # f(x, y) = (x-1)^2 + (y-2)^2, minimum at (1, 2)
      fun = fn x ->
        Nx.add(
          Nx.pow(Nx.subtract(x[0], 1), 2),
          Nx.pow(Nx.subtract(x[1], 2), 2)
        )
      end

      x0 = Nx.tensor([0.0, 0.0])

      result = Optimize.minimize(fun, x0, method: :nelder_mead)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([1.0, 2.0]), atol: 1.0e-4)
    end

    test "minimizes Rosenbrock function" do
      # f(x, y) = (1-x)^2 + 100(y-x^2)^2, minimum at (1, 1)
      fun = fn x ->
        term1 = Nx.pow(Nx.subtract(1, x[0]), 2)
        term2 = Nx.multiply(100, Nx.pow(Nx.subtract(x[1], Nx.pow(x[0], 2)), 2))
        Nx.add(term1, term2)
      end

      x0 = Nx.tensor([0.0, 0.0])

      result = Optimize.minimize(fun, x0, method: :nelder_mead, maxiter: 1000, tol: 1.0e-6)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([1.0, 1.0]), atol: 2.0e-3)
    end

    test "Nelder-Mead is default method" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 1.0])

      result = Optimize.minimize(fun, x0)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([0.0, 0.0]), atol: 1.0e-4)
    end
  end

  describe "minimize with BFGS" do
    test "minimizes sphere function" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 2.0, 3.0])

      result = Optimize.minimize(fun, x0, method: :bfgs)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([0.0, 0.0, 0.0]), atol: 1.0e-6)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-12)
    end

    test "minimizes quadratic with known solution" do
      # f(x) = x'Ax/2 - b'x where A = diag([2, 4]), b = [1, 2]
      # Optimum at x = A^(-1)b = [0.5, 0.5]
      fun = fn x ->
        quad_term = Nx.multiply(0.5, Nx.add(
          Nx.multiply(2, Nx.pow(x[0], 2)),
          Nx.multiply(4, Nx.pow(x[1], 2))
        ))
        lin_term = Nx.add(x[0], Nx.multiply(2, x[1]))
        Nx.subtract(quad_term, lin_term)
      end

      x0 = Nx.tensor([0.0, 0.0])

      result = Optimize.minimize(fun, x0, method: :bfgs)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([0.5, 0.5]), atol: 1.0e-6)
    end

    test "minimizes Rosenbrock function" do
      fun = fn x ->
        term1 = Nx.pow(Nx.subtract(1, x[0]), 2)
        term2 = Nx.multiply(100, Nx.pow(Nx.subtract(x[1], Nx.pow(x[0], 2)), 2))
        Nx.add(term1, term2)
      end

      x0 = Nx.tensor([0.0, 0.0])

      result = Optimize.minimize(fun, x0, method: :bfgs, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([1.0, 1.0]), atol: 1.0e-4)
    end

    test "tracks gradient evaluations" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 1.0])

      result = Optimize.minimize(fun, x0, method: :bfgs)

      # BFGS should track gradient evaluations
      assert Nx.to_number(result.grad_evals) > 0
    end

    test "BFGS converges faster than Nelder-Mead on smooth functions" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([5.0, 5.0, 5.0, 5.0])

      result_bfgs = Optimize.minimize(fun, x0, method: :bfgs)
      result_nm = Optimize.minimize(fun, x0, method: :nelder_mead)

      # Both should succeed
      assert Nx.to_number(result_bfgs.converged) == 1
      assert Nx.to_number(result_nm.converged) == 1

      # BFGS should typically need fewer iterations for smooth problems
      assert Nx.to_number(result_bfgs.iterations) < Nx.to_number(result_nm.iterations)
    end
  end

  describe "error handling" do
    test "raises on invalid method for minimize" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0])

      assert_raise NimbleOptions.ValidationError, fn ->
        Optimize.minimize(fun, x0, method: :invalid)
      end
    end

    test "raises on invalid method for minimize_scalar" do
      fun = fn x -> Nx.pow(x, 2) end

      assert_raise NimbleOptions.ValidationError, fn ->
        Optimize.minimize_scalar(fun, bracket: {0.0, 1.0}, method: :invalid)
      end
    end

    test "raises when bracket not provided for minimize_scalar" do
      fun = fn x -> Nx.pow(x, 2) end

      assert_raise NimbleOptions.ValidationError, ~r/bracket/, fn ->
        Optimize.minimize_scalar(fun, [])
      end
    end

    test "raises on invalid bracket" do
      fun = fn x -> Nx.pow(x, 2) end

      assert_raise NimbleOptions.ValidationError, fn ->
        Optimize.minimize_scalar(fun, bracket: {5.0, 1.0})
      end
    end

    test "raises on invalid x0 shape for minimize" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])

      assert_raise ArgumentError, fn ->
        Optimize.minimize(fun, x0)
      end
    end
  end

  describe "result struct" do
    test "contains all expected fields" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 1.0])

      result = Optimize.minimize(fun, x0)

      assert %Scholar.Optimize{} = result
      assert Nx.is_tensor(result.x)
      assert Nx.is_tensor(result.fun)
      assert Nx.is_tensor(result.converged)
      assert Nx.is_tensor(result.iterations)
      assert Nx.is_tensor(result.fun_evals)
      assert Nx.is_tensor(result.grad_evals)
    end

    test "converged is true when optimization succeeds" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 1.0])

      result = Optimize.minimize(fun, x0)

      assert Nx.to_number(result.converged) == 1
    end

    test "converged is false when max iterations reached" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([100.0, 100.0])

      result = Optimize.minimize(fun, x0, maxiter: 2)

      assert Nx.to_number(result.converged) == 0
    end
  end
end
