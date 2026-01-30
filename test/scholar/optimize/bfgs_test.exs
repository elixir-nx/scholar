defmodule Scholar.Optimize.BFGSTest do
  use Scholar.Case, async: true
  alias Scholar.Optimize.BFGS
  doctest BFGS

  describe "minimize/3" do
    test "minimizes sphere function" do
      # f(x) = sum(x^2), minimum at origin
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 2.0, 3.0])

      result = BFGS.minimize(x0, fun, gtol: 1.0e-6, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([0.0, 0.0, 0.0]), atol: 1.0e-4)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-8)
    end

    test "minimizes Rosenbrock function" do
      # f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1, 1)
      rosenbrock = fn x ->
        x0 = x[0]
        x1 = x[1]
        term1 = Nx.pow(Nx.subtract(1, x0), 2)
        term2 = Nx.multiply(100, Nx.pow(Nx.subtract(x1, Nx.pow(x0, 2)), 2))
        Nx.add(term1, term2)
      end

      x0 = Nx.tensor([0.0, 0.0], type: :f64)

      result = BFGS.minimize(x0, rosenbrock, gtol: 1.0e-8, maxiter: 1000)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([1.0, 1.0]), atol: 1.0e-4)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-6)
    end

    test "minimizes Booth function" do
      # f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2, minimum at (1, 3)
      booth = fn x ->
        x0 = x[0]
        x1 = x[1]
        # (x + 2y - 7)^2
        term1 = Nx.pow(Nx.subtract(Nx.add(x0, Nx.multiply(2, x1)), 7), 2)
        # (2x + y - 5)^2
        term2 = Nx.pow(Nx.subtract(Nx.add(Nx.multiply(2, x0), x1), 5), 2)
        Nx.add(term1, term2)
      end

      x0 = Nx.tensor([0.0, 0.0])

      result = BFGS.minimize(x0, booth, gtol: 1.0e-6, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([1.0, 3.0]), atol: 1.0e-4)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-8)
    end

    test "minimizes Beale function" do
      # f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
      # minimum at (3, 0.5)
      beale = fn x ->
        x0 = x[0]
        x1 = x[1]

        # 1.5 - x + xy
        t1 = Nx.add(Nx.subtract(1.5, x0), Nx.multiply(x0, x1))
        term1 = Nx.pow(t1, 2)

        # 2.25 - x + xy^2
        t2 = Nx.add(Nx.subtract(2.25, x0), Nx.multiply(x0, Nx.pow(x1, 2)))
        term2 = Nx.pow(t2, 2)

        # 2.625 - x + xy^3
        t3 = Nx.add(Nx.subtract(2.625, x0), Nx.multiply(x0, Nx.pow(x1, 3)))
        term3 = Nx.pow(t3, 2)

        Nx.add(Nx.add(term1, term2), term3)
      end

      x0 = Nx.tensor([0.0, 0.0], type: :f64)

      result = BFGS.minimize(x0, beale, gtol: 1.0e-8, maxiter: 1000)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([3.0, 0.5]), atol: 1.0e-3)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-6)
    end

    test "minimizes shifted quadratic" do
      # f(x) = (x1 - 2)^2 + (x2 + 3)^2 + (x3 - 1)^2, minimum at (2, -3, 1)
      fun = fn x ->
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        term1 = Nx.pow(Nx.subtract(x0, 2), 2)
        term2 = Nx.pow(Nx.add(x1, 3), 2)
        term3 = Nx.pow(Nx.subtract(x2, 1), 2)
        Nx.add(Nx.add(term1, term2), term3)
      end

      x0 = Nx.tensor([0.0, 0.0, 0.0])

      result = BFGS.minimize(x0, fun, gtol: 1.0e-6, maxiter: 500)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([2.0, -3.0, 1.0]), atol: 1.0e-4)
      assert_all_close(result.fun, Nx.tensor(0.0), atol: 1.0e-8)
    end

    test "respects maxiter limit" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([100.0, 100.0])

      result = BFGS.minimize(x0, fun, gtol: 1.0e-15, maxiter: 5)

      assert Nx.to_number(result.iterations) <= 5
    end

    test "works with jit_apply" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 2.0])
      opts = [gtol: 1.0e-6, maxiter: 500]

      result = Nx.Defn.jit_apply(&BFGS.minimize/3, [x0, fun, opts])

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([0.0, 0.0]), atol: 1.0e-4)
    end

    test "returns correct struct" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 2.0])

      result = BFGS.minimize(x0, fun, gtol: 1.0e-5, maxiter: 500)

      assert %Scholar.Optimize.BFGS{} = result
      assert Map.has_key?(result, :grad_evals)
    end

    test "achieves higher precision with f64" do
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 2.0], type: :f64)

      result = BFGS.minimize(x0, fun, gtol: 1.0e-12, maxiter: 1000)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([0.0, 0.0]), atol: 1.0e-10)
    end

    test "handles higher dimensions" do
      # 5-dimensional sphere function
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      result = BFGS.minimize(x0, fun, gtol: 1.0e-5, maxiter: 1000)

      assert Nx.to_number(result.converged) == 1
      assert_all_close(result.x, Nx.tensor([0.0, 0.0, 0.0, 0.0, 0.0]), atol: 1.0e-3)
    end

    test "converges faster than Nelder-Mead on smooth functions" do
      # BFGS should use significantly fewer function evaluations
      fun = fn x -> Nx.sum(Nx.pow(x, 2)) end
      x0 = Nx.tensor([1.0, 2.0, 3.0])

      bfgs_result = BFGS.minimize(x0, fun, gtol: 1.0e-6)
      nm_result = Scholar.Optimize.NelderMead.minimize(x0, fun, tol: 1.0e-6)

      # Both should converge
      assert Nx.to_number(bfgs_result.converged) == 1
      assert Nx.to_number(nm_result.converged) == 1

      # BFGS should use fewer function evaluations for this smooth problem
      assert Nx.to_number(bfgs_result.fun_evals) < Nx.to_number(nm_result.fun_evals)
    end
  end
end
