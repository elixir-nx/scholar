defmodule Scholar.Interpolation.CubicSplineTest do
  use Scholar.Case, async: true
  import Nx, only: :sigils

  alias Scholar.Interpolation.CubicSpline
  doctest CubicSpline

  describe "cubic spline" do
    test "fit/2" do
      # Reference values taken from Scipy
      x = Nx.iota({5})
      y = Nx.tensor([1, 2, 3, -10, -1])

      model = CubicSpline.fit(x, y)

      assert_all_close(
        model.coefficients,
        Nx.tensor([
          [-4.41666841506958, 13.250003814697266, -7.833336353302002, 1.0],
          [-4.416666030883789, -1.9073486328125e-6, 5.416667938232422, 2.0],
          [8.083332061767578, -13.249998092651367, -7.833333969116211, 3.0],
          [8.083335876464844, 10.999996185302734, -10.083333015441895, -10.0]
        ])
      )
    end

    test "train/2 bc=natural" do
      # Reference values taken from Scipy
      x = Nx.iota({5})
      y = Nx.tensor([1, 2, 3, -10, -1])

      model = CubicSpline.fit(x, y, boundary_condition: :natural)

      assert_all_close(
        model.coefficients,
        Nx.tensor([
          [1.3928568363189697, 3.5762786865234375e-7, -0.392857164144516, 1.0],
          [-6.964285850524902, 4.178571701049805, 3.7857139110565186, 2.0],
          [12.464284896850586, -16.714284896850586, -8.75, 3.0],
          [-6.892856597900391, 20.678571701049805, -4.785714626312256, -10.0]
        ])
      )
    end

    test "input validation error cases" do
      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 2, got: {1, 1, 1}",
                   fn ->
                     CubicSpline.fit(Nx.iota({1, 1, 1}), Nx.iota({1, 1, 1}))
                   end

      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 2, got: {}",
                   fn ->
                     CubicSpline.fit(Nx.iota({}), Nx.iota({}))
                   end

      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 2, got: {2}",
                   fn ->
                     CubicSpline.fit(Nx.iota({2}), Nx.iota({2}))
                   end

      assert_raise ArgumentError, "expected y to have shape {4}, got: {3}", fn ->
        CubicSpline.fit(Nx.iota({4}), Nx.iota({3}))
      end
    end

    test "predict/2" do
      # Reference values taken from Scipy
      x = Nx.iota({5})
      y = Nx.tensor([1, 2, 3, -10, -1])

      model = CubicSpline.fit(x, y)

      # ensure given values are predicted accurately
      # also ensures that the code works for scalar tensors
      assert_all_close(CubicSpline.predict(model, 0), Nx.tensor(1.0))
      assert_all_close(CubicSpline.predict(model, 1), Nx.tensor(2.0))
      assert_all_close(CubicSpline.predict(model, 2), Nx.tensor(3.0))
      assert_all_close(CubicSpline.predict(model, 3), Nx.tensor(-10.0))
      assert_all_close(CubicSpline.predict(model, 4), Nx.tensor(-1.0))

      # Test for continuity over the given point's boundaries
      # (helps ensure no off-by-one's are happening when selecting polynomials)
      assert_all_close(
        CubicSpline.predict(
          model,
          Nx.tensor([-0.001, 0.001, 0.999, 1.001, 1.999, 2.001, 2.999, 3.001, 3.999, 4.001])
        ),
        Nx.tensor([
          1.0078465938568115,
          0.9921799302101135,
          1.9945827722549438,
          2.0054168701171875,
          3.0078206062316895,
          2.9921538829803467,
          -9.989906311035156,
          -10.010071754455566,
          -1.0361297130584717,
          -0.9638023376464844
        ])
      )

      # ensure reference values are calculated accordingly
      x_predict = Nx.tensor([-1, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5])

      assert_all_close(
        CubicSpline.predict(model, x_predict),
        Nx.tensor([
          26.50000762939453,
          8.78125286102295,
          -0.15625077486038208,
          4.15625,
          -3.21875,
          -11.28125,
          26.90625,
          78.50000762939453
        ])
      )
    end

    test "predict/2 bc=natural" do
      # Reference values taken from Scipy
      x = Nx.iota({5})
      y = Nx.tensor([1, 2, 3, -10, -1])

      model = CubicSpline.fit(x, y, boundary_condition: :natural)

      # ensure given values are predicted accurately
      # also ensures that the code works for scalar tensors
      assert_all_close(CubicSpline.predict(model, 0), Nx.tensor(1.0))
      assert_all_close(CubicSpline.predict(model, 1), Nx.tensor(2.0))
      assert_all_close(CubicSpline.predict(model, 2), Nx.tensor(3.0))
      assert_all_close(CubicSpline.predict(model, 3), Nx.tensor(-10.0))
      assert_all_close(CubicSpline.predict(model, 4), Nx.tensor(-1.0))

      # Test for continuity over the given point's boundaries
      # (helps ensure no off-by-one's are happening when selecting polynomials)
      assert_all_close(
        CubicSpline.predict(
          model,
          Nx.tensor([-0.001, 0.001, 0.999, 1.001, 1.999, 2.001, 2.999, 3.001, 3.999, 4.001])
        ),
        Nx.tensor([
          1.00039286,
          0.99960714,
          1.9962185621261597,
          2.0037901401519775,
          3.0087337493896484,
          2.9912338256835938,
          -9.995194435119629,
          -10.004764556884766,
          -1.0158908367156982,
          -0.9841086268424988
        ])
      )

      # ensure reference values are calculated accordingly
      x_predict = Nx.tensor([-1, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5])

      assert_all_close(
        CubicSpline.predict(model, x_predict),
        Nx.tensor([
          6.854534149169922e-7,
          1.0223215818405151,
          0.97767857,
          4.06696429,
          -3.99553571,
          -8.08482143,
          6.0848236083984375,
          8.000004768371582
        ])
      )
    end

    test "predict/2 returns NaN if out-of-bounds and extrapolate: false" do
      model = CubicSpline.fit(Nx.tensor([0, 1, 2]), Nx.tensor([0, 1, 2]))

      assert CubicSpline.predict(model, Nx.tensor([-1, 0, 1, 2, 3]), extrapolate: false) ==
               ~V[NaN 0 1 2 NaN]
    end

    test "predict/2 returns the same shape as the input" do
      # train a straight line
      x = y = Nx.iota({3})

      model = CubicSpline.fit(x, y)

      # ensure the coefficients correspont to a straight line
      # we can see this because each line corresponds to
      # [a, b, c, d] such that f(x) = a*x**3 + b*x**2 + c*x + d
      # since a and b are both zero (or close), we have f(x) = c*x + d
      assert_all_close(
        model.coefficients,
        Nx.tensor([
          [0.0, 5.960464477539063e-8, 0.9999999403953552, 0.0],
          [0.0, 0.0, 1.0, 1.0]
        ])
      )

      for shape <- [{}, {2}, {3, 4}, {5, 6, 7}] do
        target_x = Nx.iota(shape, type: :f32)
        # a straight line will output the input
        assert CubicSpline.predict(model, target_x) == target_x
      end
    end

    test "not sorted x" do
      x = Nx.tensor([3, 2, 4, 1, 0])
      y = Nx.tensor([-10, 3, -1, 2, 1])

      model = CubicSpline.fit(x, y)

      # ensure given values are predicted accurately
      # also ensures that the code works for scalar tensors
      assert_all_close(CubicSpline.predict(model, 0), Nx.tensor(1.0))
      assert_all_close(CubicSpline.predict(model, 1), Nx.tensor(2.0))
      assert_all_close(CubicSpline.predict(model, 2), Nx.tensor(3.0))
      assert_all_close(CubicSpline.predict(model, 3), Nx.tensor(-10.0))
      assert_all_close(CubicSpline.predict(model, 4), Nx.tensor(-1.0))

      # Test for continuity over the given point's boundaries
      # (helps ensure no off-by-one's are happening when selecting polynomials)
      assert_all_close(
        CubicSpline.predict(
          model,
          Nx.tensor([-0.001, 0.001, 0.999, 1.001, 1.999, 2.001, 2.999, 3.001, 3.999, 4.001])
        ),
        Nx.tensor([
          1.0078465938568115,
          0.9921799302101135,
          1.9945827722549438,
          2.0054168701171875,
          3.0078206062316895,
          2.9921538829803467,
          -9.989906311035156,
          -10.010071754455566,
          -1.0361297130584717,
          -0.9638023376464844
        ])
      )

      # ensure reference values are calculated accordingly
      x_predict = Nx.tensor([-1, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5])

      assert_all_close(
        CubicSpline.predict(model, x_predict),
        Nx.tensor([
          26.50000762939453,
          8.78125286102295,
          -0.15625077486038208,
          4.15625,
          -3.21875,
          -11.28125,
          26.90625,
          78.50000762939453
        ])
      )
    end
  end
end
