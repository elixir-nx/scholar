defmodule Scholar.Interpolation.BezierSplineTest do
  use Scholar.Case, async: true
  alias Scholar.Interpolation.BezierSpline
  doctest BezierSpline

  describe "bezier spline" do
    test "fit/2" do
      x = Nx.iota({4})
      y = Nx.tensor([0, 1, 8, 1])

      model = BezierSpline.fit(x, y)

      assert_all_close(
        model.coefficients,
        Nx.tensor([
          [
            [0.0, 0.0],
            [0.3333335816860199, -0.5111109614372253],
            [0.6666669845581055, -1.0222218036651611],
            [1.0, 1.0]
          ],
          [
            [1.0, 1.0],
            [1.3333330154418945, 3.022221803665161],
            [1.6666665077209473, 7.577777862548828],
            [2.0, 8.0]
          ],
          [
            [2.0, 8.0],
            [2.3333334922790527, 8.422222137451172],
            [2.6666667461395264, 4.711111068725586],
            [3.0, 1.0]
          ]
        ])
      )

      assert model.k == Nx.stack([x, y], axis: 1)
    end

    test "input validation error cases" do
      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 2, got: {1, 1, 1}",
                   fn ->
                     BezierSpline.fit(Nx.iota({1, 1, 1}), Nx.iota({1, 1, 1}))
                   end

      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 2, got: {}",
                   fn ->
                     BezierSpline.fit(Nx.iota({}), Nx.iota({}))
                   end

      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 2, got: {1}",
                   fn ->
                     BezierSpline.fit(Nx.iota({1}), Nx.iota({1}))
                   end

      assert_raise ArgumentError, "expected y to have shape {4}, got: {3}", fn ->
        BezierSpline.fit(Nx.iota({4}), Nx.iota({3}))
      end
    end

    test "predict/2" do
      x = Nx.iota({4})
      y = Nx.tensor([0, 1, 8, 1])

      model = BezierSpline.fit(x, y)

      assert_all_close(
        BezierSpline.predict(model, Nx.tensor([0, 1, 2, 3, -0.5, 0.5, 1.5, 2.5, 3.5]),
          max_iter: 20,
          eps: 1.0e-3
        ),
        Nx.tensor([
          0.0,
          0.9881857633590698,
          7.997480392456055,
          1.0217341184616089,
          7.31151374111505e-7,
          -0.4500003159046173,
          5.083068370819092,
          6.065662860870361,
          -4.0382304191589355
        ])
      )
    end
  end
end
