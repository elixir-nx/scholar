defmodule Scholar.Optimize.BrentTest do
  use Scholar.Case, async: true
  alias Scholar.Optimize.Brent
  doctest Brent

  describe "minimize/4" do
    test "minimizes simple parabola" do
      # f(x) = (x - 3)^2, minimum at x = 3
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result = Brent.minimize(0.0, 5.0, fun, tol: 1.0e-5, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(3.0), atol: 1.0e-4)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-8)
    end

    test "minimizes shifted parabola" do
      # f(x) = (x + 2)^2 + 1, minimum at x = -2, f(-2) = 1
      fun = fn x -> Nx.add(Nx.pow(Nx.add(x, 2), 2), 1) end

      result = Brent.minimize(-5.0, 5.0, fun, tol: 1.0e-5, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(-2.0), atol: 1.0e-4)
      assert_all_close(result.fun, Nx.tensor(1.0), atol: 1.0e-8)
    end

    test "minimizes sine function" do
      # Minimum of sin(x) in [3, 5] is at x = 3*pi/2 ≈ 4.712
      fun = fn x -> Nx.sin(x) end
      expected_x = 3 * :math.pi() / 2

      result = Brent.minimize(3.0, 5.0, fun, tol: 1.0e-5, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(expected_x), atol: 1.0e-4)
      assert_all_close(result.fun, Nx.tensor(-1.0), atol: 1.0e-6)
    end

    test "respects maxiter limit" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result = Brent.minimize(0.0, 100.0, fun, tol: 1.0e-5, maxiter: 5)

      assert Nx.to_number(result.iterations) <= 5
    end

    test "works with jit_apply" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end
      opts = [tol: 1.0e-5, maxiter: 500]

      result = Nx.Defn.jit_apply(&Brent.minimize/4, [0.0, 5.0, fun, opts])

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(3.0), atol: 1.0e-4)
    end

    test "returns correct struct" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result = Brent.minimize(0.0, 5.0, fun, tol: 1.0e-5, maxiter: 500)

      assert %Scholar.Optimize.Brent{} = result
    end

    test "works with tensor bounds" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end
      a = Nx.tensor(0.0)
      b = Nx.tensor(5.0)

      result = Brent.minimize(a, b, fun)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(3.0), atol: 1.0e-4)
    end

    test "achieves higher precision with f64 bounds" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end
      a = Nx.tensor(0.0, type: :f64)
      b = Nx.tensor(5.0, type: :f64)

      result = Brent.minimize(a, b, fun, tol: 1.0e-10)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(3.0), atol: 1.0e-8)
    end

    test "converges faster than golden section" do
      # Brent should use significantly fewer function evaluations
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      brent_result = Brent.minimize(0.0, 5.0, fun, tol: 1.0e-5, maxiter: 500)

      golden_result =
        Scholar.Optimize.GoldenSection.minimize(0.0, 5.0, fun, tol: 1.0e-5, maxiter: 500)

      # Both should converge
      assert Nx.to_number(brent_result.converged) == 1
      assert Nx.to_number(golden_result.converged) == 1

      # Brent should use fewer function evaluations
      # SciPy reference: Brent ~8, Golden ~45
      assert Nx.to_number(brent_result.fun_evals) < Nx.to_number(golden_result.fun_evals)
    end

    test "handles wide bracket" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 50), 2) end

      result = Brent.minimize(0.0, 100.0, fun, tol: 1.0e-5, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(50.0), atol: 1.0e-3)
    end

    test "handles asymmetric function" do
      # f(x) = x^4 - 2*x^2, minima near x = ±1
      fun = fn x -> Nx.subtract(Nx.pow(x, 4), Nx.multiply(2, Nx.pow(x, 2))) end

      # Search in positive region
      result = Brent.minimize(0.0, 2.0, fun, tol: 1.0e-5, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(1.0), atol: 1.0e-3)
    end
  end
end
