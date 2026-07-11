defmodule Scholar.Interpolation.MonotonicCubicSplineTest do
  use Scholar.Case, async: true
  import Nx, only: :sigils

  alias Scholar.Interpolation.MonotonicCubicSpline
  doctest MonotonicCubicSpline

  describe "monotonic cubic spline" do
    test "fit/2" do
      # Reference values taken from SciPy (scipy.interpolate.PchipInterpolator)
      x = Nx.iota({5})
      y = Nx.tensor([1, 2, 3, -10, -1])

      model = MonotonicCubicSpline.fit(x, y)

      assert_all_close(
        model.coefficients,
        Nx.tensor([
          [0.0, 0.0, 1.0, 1.0],
          [-1.0, 1.0, 1.0, 2.0],
          [26.0, -39.0, 0.0, 3.0],
          [2.0, 7.0, 0.0, -10.0]
        ])
      )
    end

    test "fit/2 with only two points" do
      x = Nx.tensor([0, 1])
      y = Nx.tensor([0.0, 2.0])

      model = MonotonicCubicSpline.fit(x, y)

      # a single interval collapses to a straight line
      assert_all_close(model.coefficients, Nx.tensor([[0.0, 0.0, 2.0, 0.0]]))
      assert_all_close(MonotonicCubicSpline.predict(model, Nx.tensor(0.5)), Nx.tensor(1.0))
    end

    test "input validation error cases" do
      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 1, got: {1, 1, 1}",
                   fn ->
                     MonotonicCubicSpline.fit(Nx.iota({1, 1, 1}), Nx.iota({1, 1, 1}))
                   end

      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 1, got: {}",
                   fn ->
                     MonotonicCubicSpline.fit(Nx.iota({}), Nx.iota({}))
                   end

      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 1, got: {1}",
                   fn ->
                     MonotonicCubicSpline.fit(Nx.iota({1}), Nx.iota({1}))
                   end

      assert_raise ArgumentError, "expected y to have shape {4}, got: {3}", fn ->
        MonotonicCubicSpline.fit(Nx.iota({4}), Nx.iota({3}))
      end
    end

    test "predict/2" do
      # Reference values taken from SciPy (scipy.interpolate.PchipInterpolator)
      x = Nx.iota({5})
      y = Nx.tensor([1, 2, 3, -10, -1])

      model = MonotonicCubicSpline.fit(x, y)

      # ensure given values are predicted accurately
      # also ensures that the code works for scalar tensors
      assert_all_close(MonotonicCubicSpline.predict(model, 0), Nx.tensor(1.0))
      assert_all_close(MonotonicCubicSpline.predict(model, 1), Nx.tensor(2.0))
      assert_all_close(MonotonicCubicSpline.predict(model, 2), Nx.tensor(3.0))
      assert_all_close(MonotonicCubicSpline.predict(model, 3), Nx.tensor(-10.0))
      assert_all_close(MonotonicCubicSpline.predict(model, 4), Nx.tensor(-1.0))

      # Test for continuity over the given point's boundaries
      # (helps ensure no off-by-one's are happening when selecting polynomials)
      assert_all_close(
        MonotonicCubicSpline.predict(
          model,
          Nx.tensor([-0.001, 0.001, 0.999, 1.001, 1.999, 2.001, 2.999, 3.001, 3.999, 4.001])
        ),
        Nx.tensor([
          0.999,
          1.001,
          1.999,
          2.001000999,
          2.999998001,
          2.999961026,
          -9.999961026,
          -9.999992998,
          -1.019987002,
          -0.979986998
        ])
      )

      # ensure reference values are calculated accordingly
      x_predict = Nx.tensor([-1, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5])

      assert_all_close(
        MonotonicCubicSpline.predict(model, x_predict),
        Nx.tensor([0.0, 0.5, 1.5, 2.625, -3.5, -8.0, 12.5, 34.0])
      )
    end

    test "predict/2 preserves monotonicity (no overshoot)" do
      # For monotonic data the interpolant must stay monotonic, unlike a
      # regular cubic spline which can overshoot around the flat region.
      x = Nx.iota({5})
      y = Nx.tensor([0.0, 1.0, 1.0, 1.0, 5.0])

      model = MonotonicCubicSpline.fit(x, y)

      dense = Nx.linspace(0.0, 4.0, n: 81)
      values = MonotonicCubicSpline.predict(model, dense)

      # non-decreasing everywhere
      assert Nx.reduce_min(Nx.diff(values)) |> Nx.to_number() >= -1.0e-6

      # the flat region [1, 3] never rises above the sampled value of 1.0
      in_flat = Nx.logical_and(Nx.greater_equal(dense, 1.0), Nx.less_equal(dense, 3.0))
      max_in_flat = Nx.select(in_flat, values, 0.0) |> Nx.reduce_max() |> Nx.to_number()
      assert max_in_flat <= 1.0 + 1.0e-6
    end

    test "predict/2 returns NaN if out-of-bounds and extrapolate: false" do
      model = MonotonicCubicSpline.fit(Nx.tensor([0, 1, 2]), Nx.tensor([0, 1, 2]))

      assert MonotonicCubicSpline.predict(model, Nx.tensor([-1, 0, 1, 2, 3]), extrapolate: false) ==
               ~VEC[NaN 0 1 2 NaN]
    end

    test "predict/2 returns the same shape as the input" do
      # train a straight line
      x = y = Nx.iota({3})

      model = MonotonicCubicSpline.fit(x, y)

      # a straight line has zero second/third order coefficients
      assert_all_close(
        model.coefficients,
        Nx.tensor([
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0, 1.0]
        ])
      )

      for shape <- [{}, {2}, {3, 4}, {5, 6, 7}] do
        target_x = Nx.iota(shape, type: :f32)
        # a straight line will output the input
        assert MonotonicCubicSpline.predict(model, target_x) == target_x
      end
    end

    test "fit/2 and predict/2 work with jit_apply" do
      x = Nx.iota({5})
      y = Nx.tensor([1, 2, 3, -10, -1])

      model = Nx.Defn.jit_apply(&MonotonicCubicSpline.fit/2, [x, y])

      prediction =
        Nx.Defn.jit_apply(&MonotonicCubicSpline.predict/3, [model, Nx.tensor(1.0), []])

      assert prediction == Nx.tensor(2.0)
    end

    test "fit/2 propagates input precision (f64)" do
      # f64 inputs must not be downcast to f32
      x = Nx.tensor([0, 1, 2, 3, 4], type: :f64)
      y = Nx.tensor([1, 2, 3, -10, -1], type: :f64)

      model = MonotonicCubicSpline.fit(x, y)
      assert Nx.type(model.coefficients) == {:f, 64}

      prediction = MonotonicCubicSpline.predict(model, Nx.tensor([0.5, 2.5], type: :f64))
      assert Nx.type(prediction) == {:f, 64}
      assert_all_close(prediction, Nx.tensor([1.5, -3.5], type: :f64))
    end

    test "not sorted x" do
      x = Nx.tensor([3, 2, 4, 1, 0])
      y = Nx.tensor([-10, 3, -1, 2, 1])

      model = MonotonicCubicSpline.fit(x, y)

      # sorting recovers the same points as the "predict/2" test
      assert_all_close(MonotonicCubicSpline.predict(model, 0), Nx.tensor(1.0))
      assert_all_close(MonotonicCubicSpline.predict(model, 1), Nx.tensor(2.0))
      assert_all_close(MonotonicCubicSpline.predict(model, 2), Nx.tensor(3.0))
      assert_all_close(MonotonicCubicSpline.predict(model, 3), Nx.tensor(-10.0))
      assert_all_close(MonotonicCubicSpline.predict(model, 4), Nx.tensor(-1.0))

      x_predict = Nx.tensor([-1, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5])

      assert_all_close(
        MonotonicCubicSpline.predict(model, x_predict),
        Nx.tensor([0.0, 0.5, 1.5, 2.625, -3.5, -8.0, 12.5, 34.0])
      )
    end
  end
end
