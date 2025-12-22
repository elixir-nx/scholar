defmodule Scholar.Optimize.GoldenSectionTest do
  use Scholar.Case, async: true
  alias Scholar.Optimize.GoldenSection
  doctest GoldenSection

  describe "minimize/2" do
    test "minimizes simple parabola" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result = GoldenSection.minimize(fun, bracket: {0.0, 5.0}, tol: 1.0e-8, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(3.0), atol: 1.0e-6)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-10)
    end

    test "minimizes shifted parabola" do
      fun = fn x -> Nx.add(Nx.pow(Nx.add(x, 2), 2), 1) end

      result = GoldenSection.minimize(fun, bracket: {-5.0, 5.0}, tol: 1.0e-8, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(-2.0), atol: 1.0e-6)
      assert_all_close(result.fun, Nx.tensor(1.0), atol: 1.0e-10)
    end

    test "minimizes sine function" do
      # Minimum of sin(x) in [0, 2*pi] is at x = 3*pi/2
      fun = fn x -> Nx.sin(x) end
      expected_x = 3 * :math.pi() / 2

      result = GoldenSection.minimize(fun, bracket: {0.0, 2 * :math.pi()}, tol: 1.0e-8, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(expected_x), atol: 1.0e-5)
    end

    test "respects maxiter limit" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result = GoldenSection.minimize(fun, bracket: {0.0, 100.0}, tol: 1.0e-8, maxiter: 5)

      assert Nx.to_number(result.iterations) <= 5
    end

    test "works with jit_apply" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end
      opts = [bracket: {0.0, 5.0}, tol: 1.0e-8, maxiter: 500]

      result = Nx.Defn.jit_apply(&GoldenSection.minimize/2, [fun, opts])

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor(3.0), atol: 1.0e-6)
    end

    test "returns correct struct" do
      fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end

      result = GoldenSection.minimize(fun, bracket: {0.0, 5.0}, tol: 1.0e-8, maxiter: 500)

      assert %Scholar.Optimize.GoldenSection{} = result
    end
  end
end
