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
          0.9911295771598816,
          7.998117923736572,
          1.0163085460662842,
          0.4502662420272827,
          -0.4506153464317322,
          5.102115631103516,
          6.063710689544678,
          -4.0441107749938965
        ])
      )
    end
  end
end
