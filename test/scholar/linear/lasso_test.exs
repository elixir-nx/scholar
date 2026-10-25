defmodule Scholar.Linear.LassoTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.{ElasticNet, Lasso}
  doctest Lasso

  # scikit-learn 1.6.1's `sklearn.linear_model.Lasso` is the reference, fitted on the
  # same data with `precompute=False` (the default); the iteration counts below are its
  # own. The f64 expectations are its coefficients to full precision. The f32 ones are
  # this implementation's, which agree with it to the tolerance on each assertion.

  describe "fit" do
    test "matches scikit-learn" do
      {x, y} = diabetes_data(:f64)

      model = Lasso.fit(x, y, alpha: 10.0)

      expected_coefficients =
        Nx.tensor(
          [
            -0.36746418227743394,
            0.0,
            7.349765026065425,
            0.8376782001289618,
            2.562256698081846,
            -3.057470261176257,
            -2.620645588480563,
            0.0,
            0.0,
            -1.340024966289126
          ],
          type: :f64
        )

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-10, rtol: 1.0e-10)

      assert_all_close(model.intercept, Nx.tensor(7.9489655583381875, type: :f64),
        atol: 1.0e-10,
        rtol: 1.0e-10
      )

      assert Nx.to_number(model.num_iterations) == 93

      assert_all_close(model.dual_gap, Nx.tensor(0.49904933057376183, type: :f64),
        atol: 1.0e-10,
        rtol: 1.0e-8
      )
    end

    test "a larger alpha keeps fewer features" do
      {x, y} = diabetes_data()

      model = Lasso.fit(x, y, alpha: 100.0)

      expected_coefficients =
        Nx.tensor([
          0.0,
          0.0,
          1.2107826471328735,
          0.895043671131134,
          1.060964584350586,
          -1.3265762329101562,
          -1.724993348121643,
          0.0,
          0.0,
          0.0
        ])

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-4, rtol: 1.0e-4)
      assert_all_close(model.intercept, Nx.tensor(67.52740478515625), atol: 1.0e-3, rtol: 1.0e-4)
      assert Nx.to_number(model.num_iterations) == 59

      nonzero = model.coefficients |> Nx.not_equal(0) |> Nx.sum() |> Nx.to_number()
      assert nonzero == 5
    end

    test "is the elastic net with l1_ratio: 1.0" do
      {x, y} = diabetes_data()
      opts = [alpha: 10.0, fit_intercept?: false, max_iterations: 50, tol: 1.0e-3]

      model = Lasso.fit(x, y, opts)
      elastic_net = ElasticNet.fit(x, y, [l1_ratio: 1.0] ++ opts)

      assert model.coefficients == elastic_net.coefficients
      assert model.intercept == elastic_net.intercept
      assert model.dual_gap == elastic_net.dual_gap
      assert model.num_iterations == elastic_net.num_iterations
    end

    test "works with jit_apply" do
      {x, y} = diabetes_data()
      opts = [alpha: 10.0]

      model = Nx.Defn.jit_apply(&Lasso.fit/3, [x, y, opts])
      expected = Lasso.fit(x, y, opts)

      assert_all_close(model.coefficients, expected.coefficients, atol: 1.0e-5, rtol: 1.0e-5)
      assert_all_close(model.intercept, expected.intercept, atol: 1.0e-4, rtol: 1.0e-5)
      assert Nx.to_number(model.num_iterations) == Nx.to_number(expected.num_iterations)

      predictions = Nx.Defn.jit_apply(&Lasso.predict/2, [model, x])
      assert_all_close(predictions, Lasso.predict(expected, x), atol: 1.0e-3, rtol: 1.0e-5)
    end

    test "f64 inputs match scikit-learn to full precision" do
      {x, y} = diabetes_data(:f64)

      model = Lasso.fit(x, y, alpha: 1.0)

      expected_coefficients =
        Nx.tensor(
          [
            -0.22149984334019215,
            -21.86511760513982,
            6.376097229661149,
            1.0008964141101515,
            0.18813654090892243,
            -0.9517313057338804,
            0.08342584433404865,
            13.06157036118588,
            69.47575892627387,
            -1.2048088130793775
          ],
          type: :f64
        )

      assert Nx.type(model.coefficients) == {:f, 64}
      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-10, rtol: 1.0e-10)

      assert_all_close(model.intercept, Nx.tensor(-270.83658761179174, type: :f64),
        atol: 1.0e-10,
        rtol: 1.0e-10
      )

      assert Nx.to_number(model.num_iterations) == 344
    end
  end

  describe "fit with positive?: true" do
    test "f64 inputs match scikit-learn to full precision" do
      {x, y} = diabetes_data(:f64)

      model = Lasso.fit(x, y, alpha: 1.0, positive?: true)

      expected_coefficients =
        Nx.tensor(
          [
            0.0,
            0.0,
            5.036420844380907,
            0.19632831356420247,
            0.0,
            0.0,
            0.0,
            0.0,
            75.0819048381021,
            0.0
          ],
          type: :f64
        )

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-10, rtol: 1.0e-10)

      assert_all_close(model.intercept, Nx.tensor(-349.19227950758795, type: :f64),
        atol: 1.0e-10,
        rtol: 1.0e-10
      )

      assert Nx.to_number(model.num_iterations) == 10

      assert_all_close(model.dual_gap, Nx.tensor(0.18248233268182956, type: :f64),
        atol: 1.0e-10,
        rtol: 1.0e-8
      )
    end
  end

  describe "fit with :sample_weights" do
    test "matches scikit-learn" do
      {x, y} = diabetes_data()
      weights = Nx.iota({50}) |> Nx.remainder(3) |> Nx.add(1)

      model = Lasso.fit(x, y, alpha: 10.0, sample_weights: weights)

      expected_coefficients =
        Nx.tensor([
          -0.24204550683498383,
          0.0,
          8.516009330749512,
          1.4087196588516235,
          1.5237725973129272,
          -2.3194687366485596,
          -1.757298231124878,
          0.0,
          0.0,
          -0.761890172958374
        ])

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-4, rtol: 1.0e-4)
      assert_all_close(model.intercept, Nx.tensor(-65.58184814453125), atol: 1.0e-3, rtol: 1.0e-4)
      assert Nx.to_number(model.num_iterations) == 90
    end
  end

  describe "errors" do
    test "does not accept :l1_ratio" do
      {x, y} = diabetes_data()

      assert_raise NimbleOptions.ValidationError, ~r/unknown options \[:l1_ratio\]/, fn ->
        Lasso.fit(x, y, l1_ratio: 0.5)
      end
    end

    test "negative :alpha" do
      {x, y} = diabetes_data()

      assert_raise NimbleOptions.ValidationError, fn -> Lasso.fit(x, y, alpha: -1.0) end
    end
  end
end
