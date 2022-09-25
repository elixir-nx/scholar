defmodule Scholar.Interpolation.CubicSplineTest do
  use ExUnit.Case, async: true

  import Nx, only: :sigils

  alias Scholar.Interpolation.CubicSpline

  describe "cubic spline" do
    test "train/2" do
      # Reference values taken from Scipy
      x = Nx.iota({5})
      y = Nx.tensor([1, 2, 3, -10, -1])

      model = CubicSpline.train(x, y)

      assert model.coefficients ==
               Nx.tensor([
                 [-4.41666841506958, 13.250003814697266, -7.833336353302002, 1.0],
                 [-4.416666030883789, -1.9073486328125e-6, 5.416667938232422, 2.0],
                 [8.083332061767578, -13.249998092651367, -7.833333969116211, 3.0],
                 [8.083335876464844, 10.999996185302734, -10.083333015441895, -10.0]
               ])
    end

    test "predict/2" do
      # Reference values taken from Scipy
      x = Nx.iota({5})
      y = Nx.tensor([1, 2, 3, -10, -1])

      model = CubicSpline.train(x, y)

      # ensure given values are predicted accurately
      # also ensures that the code works for scalar tensors
      assert CubicSpline.predict(model, 0) == Nx.tensor([1.0])
      assert CubicSpline.predict(model, 1) == Nx.tensor([1.9999990463256836])
      assert CubicSpline.predict(model, 2) == Nx.tensor([3.0])
      assert CubicSpline.predict(model, 3) == Nx.tensor([-10.0])
      assert CubicSpline.predict(model, 4) == Nx.tensor([-1.0000009536743164])

      # Test for continuity over the given point's boundaries
      # (helps ensure no off-by-one's are happening when selecting polynomials)
      assert CubicSpline.predict(
               model,
               Nx.tensor([-0.001, 0.001, 0.999, 1.001, 1.999, 2.001, 2.999, 3.001, 3.999, 4.001])
             ) ==
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

      # ensure reference values are calculated accordingly
      x_predict = Nx.tensor([-1, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5])

      assert CubicSpline.predict(model, x_predict) ==
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
    end

    test "predict/2 returns NaN if out-of-bounds and extrapolate: false" do
      model = CubicSpline.train(Nx.tensor([0, 1, 2]), Nx.tensor([0, 1, 2]))

      assert CubicSpline.predict(model, Nx.tensor([-1, 0, 1, 2, 3]), extrapolate: false) ==
               ~V[NaN 0 1 2 NaN]
    end
  end
end
