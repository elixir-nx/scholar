defmodule Scholar.Linear.ElasticNetTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.ElasticNet
  doctest ElasticNet

  # scikit-learn 1.6.1's `sklearn.linear_model.ElasticNet` is the reference, fitted on the
  # same data with `precompute=False` (the default); the iteration counts below are its
  # own. The f64 expectations are its coefficients to full precision. The f32 ones are
  # this implementation's, which agree with it to the tolerance on each assertion.

  defp small_data(type \\ :f32) do
    x =
      Nx.tensor(
        [[1.0, 2.0, 0.5], [3.0, 2.0, -1.0], [4.0, 7.0, 2.0], [0.0, -1.0, 3.0], [2.0, 5.0, 1.0]],
        type: type
      )

    y = Nx.tensor([4.0, 3.0, -1.0, 6.0, 1.0], type: type)
    {x, y}
  end

  describe "fit" do
    test "matches scikit-learn with the default options" do
      {x, y} = diabetes_data()

      model = ElasticNet.fit(x, y)

      expected_coefficients =
        Nx.tensor([
          -0.4106864631175995,
          -4.6232218742370605,
          7.5855817794799805,
          0.88670414686203,
          2.5312483310699463,
          -3.098461389541626,
          -2.4032609462738037,
          3.58559250831604,
          5.244235992431641,
          -1.5421174764633179
        ])

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-4, rtol: 1.0e-4)

      assert_all_close(model.intercept, Nx.tensor(-14.839263916015625),
        atol: 1.0e-3,
        rtol: 1.0e-4
      )

      assert Nx.to_number(model.num_iterations) == 129
      assert_all_close(model.dual_gap, Nx.tensor(0.4832937717437744), rtol: 5.0e-2)
    end

    test "matches scikit-learn with :alpha and :l1_ratio" do
      {x, y} = diabetes_data()

      model = ElasticNet.fit(x, y, alpha: 0.5, l1_ratio: 0.9)

      expected_coefficients =
        Nx.tensor([
          -0.2809448838233948,
          -18.15078353881836,
          7.05232572555542,
          0.9806502461433411,
          1.4129209518432617,
          -2.130269765853882,
          -1.0994086265563965,
          11.446634292602539,
          33.52336883544922,
          -1.3516043424606323
        ])

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-4, rtol: 1.0e-4)
      assert_all_close(model.intercept, Nx.tensor(-140.0272216796875), atol: 1.0e-3, rtol: 1.0e-4)
      assert Nx.to_number(model.num_iterations) == 245
    end

    test "l1_ratio: 1.0 sets coefficients to exactly zero" do
      {x, y} = diabetes_data(:f64)

      model = ElasticNet.fit(x, y, alpha: 10.0, l1_ratio: 1.0)

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

      for index <- [1, 7, 8] do
        assert Nx.to_number(model.coefficients[index]) == 0.0
      end
    end

    test "fit_intercept?: false" do
      {x, y} = small_data()

      model = ElasticNet.fit(x, y, alpha: 0.05, fit_intercept?: false)

      expected_coefficients =
        Nx.tensor([1.8224784135818481, -1.166018009185791, 1.2810850143432617])

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-5, rtol: 1.0e-5)
      assert Nx.to_number(model.intercept) == 0.0
      assert Nx.to_number(model.num_iterations) == 54
      assert_all_close(model.dual_gap, Nx.tensor(0.0011371185537427664), rtol: 1.0e-2)
    end

    test "a constant feature gets a zero coefficient" do
      x =
        Nx.tensor([
          [1.0, 3.0, 0.5],
          [3.0, 3.0, -1.0],
          [4.0, 3.0, 2.0],
          [0.0, 3.0, 3.0],
          [2.0, 3.0, 1.0]
        ])

      y = Nx.tensor([4.0, 3.0, -1.0, 6.0, 1.0])

      model = ElasticNet.fit(x, y, alpha: 0.05)

      expected_coefficients = Nx.tensor([-1.6104235649108887, 0.0, -0.40873801708221436])

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-5, rtol: 1.0e-5)
      assert_all_close(model.intercept, Nx.tensor(6.270459175109863), atol: 1.0e-5, rtol: 1.0e-5)
      assert Nx.to_number(model.num_iterations) == 6
      assert Nx.to_number(model.coefficients[1]) == 0.0
    end

    test ":max_iterations stops the fit early" do
      {x, y} = diabetes_data()

      model = ElasticNet.fit(x, y, alpha: 0.5, l1_ratio: 0.9, max_iterations: 5)

      expected_coefficients =
        Nx.tensor([
          -0.2592528760433197,
          -21.702308654785156,
          6.977673053741455,
          1.201479434967041,
          0.12075124680995941,
          -0.8730055093765259,
          -0.04330796003341675,
          11.435279846191406,
          54.60115432739258,
          -0.9329866170883179
        ])

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-4, rtol: 1.0e-4)
      assert_all_close(model.intercept, Nx.tensor(-242.8670654296875), atol: 1.0e-3, rtol: 1.0e-4)
      assert Nx.to_number(model.num_iterations) == 5
      assert_all_close(model.dual_gap, Nx.tensor(1124.053466796875), rtol: 1.0e-3)
    end

    test "l1_ratio: 0.0 matches ridge regression" do
      {x, y} = small_data(:f64)
      alpha = 0.2

      model = ElasticNet.fit(x, y, alpha: alpha, l1_ratio: 0.0, tol: 1.0e-12)

      # RidgeRegression minimizes ||y - Xw||^2 + alpha ||w||^2, which is the
      # elastic net objective times 2n for an alpha n times larger.
      ridge =
        Scholar.Linear.RidgeRegression.fit(x, y,
          alpha: alpha * Nx.axis_size(x, 0),
          solver: :cholesky
        )

      assert_all_close(model.coefficients, ridge.coefficients, atol: 1.0e-10, rtol: 1.0e-10)
      assert_all_close(model.intercept, ridge.intercept, atol: 1.0e-10, rtol: 1.0e-10)
    end

    test "alpha: 0.0 matches ordinary least squares" do
      {x, y} = small_data(:f64)

      model = ElasticNet.fit(x, y, alpha: 0.0, tol: 1.0e-12)

      # numpy.linalg.lstsq on the centered data
      expected_coefficients =
        Nx.tensor([-0.4149610136452235, -0.6971247563352835, -0.08674463937621812], type: :f64)

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-10, rtol: 1.0e-10)

      assert_all_close(model.intercept, Nx.tensor(5.616715399610138, type: :f64),
        atol: 1.0e-10,
        rtol: 1.0e-10
      )
    end

    test "accepts targets with shape {num_samples, 1}" do
      {x, y} = small_data()

      model = ElasticNet.fit(x, Nx.new_axis(y, 1), alpha: 0.05)
      expected = ElasticNet.fit(x, y, alpha: 0.05)

      assert model.coefficients == expected.coefficients
      assert model.intercept == expected.intercept
    end

    test "accepts integer inputs" do
      x = Nx.tensor([[1, 2], [3, 2], [4, 7]])
      y = Nx.tensor([4, 3, -1])

      model = ElasticNet.fit(x, y, alpha: 0.1)
      expected = ElasticNet.fit(Nx.as_type(x, :f32), Nx.as_type(y, :f32), alpha: 0.1)

      assert Nx.type(model.coefficients) == {:f, 32}
      assert model.coefficients == expected.coefficients
    end

    test "works with jit_apply" do
      {x, y} = small_data()
      opts = [alpha: 0.05, l1_ratio: 0.7]

      model = Nx.Defn.jit_apply(&ElasticNet.fit/3, [x, y, opts])
      expected = ElasticNet.fit(x, y, opts)

      assert_all_close(model.coefficients, expected.coefficients, atol: 1.0e-6, rtol: 1.0e-6)
      assert_all_close(model.intercept, expected.intercept, atol: 1.0e-6, rtol: 1.0e-6)
      assert Nx.to_number(model.num_iterations) == Nx.to_number(expected.num_iterations)

      predictions = Nx.Defn.jit_apply(&ElasticNet.predict/2, [model, x])
      assert_all_close(predictions, ElasticNet.predict(expected, x), atol: 1.0e-5, rtol: 1.0e-6)
    end

    test "f64 inputs match scikit-learn to full precision" do
      {x, y} = diabetes_data(:f64)

      model = ElasticNet.fit(x, y, alpha: 0.5, l1_ratio: 0.9)

      expected_coefficients =
        Nx.tensor(
          [
            -0.28094471678986765,
            -18.150803918915617,
            7.0523218336227815,
            0.9806484594312437,
            1.4129314513976976,
            -2.1302788588623183,
            -1.0994201789301488,
            11.446606121519787,
            33.52319783024136,
            -1.3516061622476911
          ],
          type: :f64
        )

      assert Nx.type(model.coefficients) == {:f, 64}
      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-10, rtol: 1.0e-10)

      assert_all_close(model.intercept, Nx.tensor(-140.02626350892515, type: :f64),
        atol: 1.0e-10,
        rtol: 1.0e-10
      )

      assert Nx.to_number(model.num_iterations) == 246

      assert_all_close(model.dual_gap, Nx.tensor(0.5221082509332337, type: :f64),
        atol: 1.0e-10,
        rtol: 1.0e-8
      )
    end
  end

  describe "fit with positive?: true" do
    test "matches scikit-learn" do
      {x, y} = diabetes_data()

      model = ElasticNet.fit(x, y, alpha: 0.5, l1_ratio: 0.9, positive?: true)

      expected_coefficients =
        Nx.tensor([
          0.0,
          0.0,
          5.843557357788086,
          0.34552234411239624,
          0.0,
          0.0,
          0.0,
          0.0,
          59.480377197265625,
          0.0
        ])

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-4, rtol: 1.0e-4)
      assert_all_close(model.intercept, Nx.tensor(-312.4940490722656), atol: 1.0e-3, rtol: 1.0e-4)
      assert Nx.to_number(model.num_iterations) == 9
      assert_all_close(model.dual_gap, Nx.tensor(0.21160589158535004), rtol: 5.0e-2)

      assert model.coefficients |> Nx.greater_equal(0) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "fit with :sample_weights" do
    # Weights 1, 2, 3, 1, 2, 3, ... over the 50 samples.
    defp diabetes_weights(type \\ :f32) do
      Nx.iota({50}, type: type) |> Nx.remainder(3) |> Nx.add(1)
    end

    test "matches scikit-learn" do
      {x, y} = diabetes_data()

      model =
        ElasticNet.fit(x, y, alpha: 0.5, l1_ratio: 0.9, sample_weights: diabetes_weights())

      expected_coefficients =
        Nx.tensor([
          -0.2730790078639984,
          -11.280350685119629,
          7.9027018547058105,
          1.4443596601486206,
          0.15722469985485077,
          -1.2773523330688477,
          0.3878490924835205,
          16.634002685546875,
          42.64410400390625,
          -0.8805441856384277
        ])

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-4, rtol: 1.0e-4)
      assert_all_close(model.intercept, Nx.tensor(-257.2607116699219), atol: 1.0e-3, rtol: 1.0e-4)

      assert Nx.to_number(model.num_iterations) == 207
    end

    test "f64 inputs match scikit-learn to full precision" do
      {x, y} = diabetes_data(:f64)

      model =
        ElasticNet.fit(x, y, alpha: 0.5, l1_ratio: 0.9, sample_weights: diabetes_weights(:f64))

      expected_coefficients =
        Nx.tensor(
          [
            -0.2730789694077994,
            -11.280362494253572,
            7.902702236458736,
            1.444359782371634,
            0.15721992579990451,
            -1.2773478196589887,
            0.3878531799506173,
            16.63401511315393,
            42.644181846950815,
            -0.8805425282225469
          ],
          type: :f64
        )

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-10, rtol: 1.0e-10)

      assert_all_close(model.intercept, Nx.tensor(-257.2610665991673, type: :f64),
        atol: 1.0e-10,
        rtol: 1.0e-10
      )

      assert Nx.to_number(model.num_iterations) == 207
    end

    test "fit_intercept?: false matches scikit-learn" do
      {x, y} = diabetes_data()

      model =
        ElasticNet.fit(x, y,
          alpha: 0.5,
          fit_intercept?: false,
          sample_weights: diabetes_weights()
        )

      expected_coefficients =
        Nx.tensor([
          -0.17961205542087555,
          -4.513019561767578,
          7.822999477386475,
          1.118497371673584,
          1.813488483428955,
          -2.6816623210906982,
          -2.1007020473480225,
          4.266779899597168,
          6.318896770477295,
          -1.4221750497817993
        ])

      assert_all_close(model.coefficients, expected_coefficients, atol: 1.0e-4, rtol: 1.0e-4)
      assert Nx.to_number(model.intercept) == 0.0
      assert Nx.to_number(model.num_iterations) == 1000
    end

    test "integer weights are the same as repeating samples" do
      {x, y} = small_data(:f64)
      weights = [1, 3, 2, 1, 2]
      repeated = Nx.tensor([0, 1, 1, 1, 2, 2, 3, 4, 4])
      opts = [alpha: 0.05, l1_ratio: 0.6, tol: 1.0e-12, max_iterations: 10_000]

      model = ElasticNet.fit(x, y, [sample_weights: weights] ++ opts)
      expected = ElasticNet.fit(Nx.take(x, repeated), Nx.take(y, repeated), opts)

      assert_all_close(model.coefficients, expected.coefficients, atol: 1.0e-8, rtol: 1.0e-8)
      assert_all_close(model.intercept, expected.intercept, atol: 1.0e-8, rtol: 1.0e-8)
    end

    test "only the ratios between weights matter" do
      {x, y} = small_data(:f64)
      opts = [alpha: 0.05, tol: 1.0e-12, max_iterations: 10_000]

      model = ElasticNet.fit(x, y, [sample_weights: Nx.tensor([2.0, 2.0, 2.0, 2.0, 2.0])] ++ opts)
      expected = ElasticNet.fit(x, y, opts)

      assert_all_close(model.coefficients, expected.coefficients, atol: 1.0e-10, rtol: 1.0e-10)
      assert_all_close(model.intercept, expected.intercept, atol: 1.0e-10, rtol: 1.0e-10)
    end

    test "works with jit_apply" do
      {x, y} = small_data()
      weights = Nx.tensor([1.0, 3.0, 2.0, 1.0, 2.0])

      model =
        Nx.Defn.jit_apply(
          fn x, y, weights -> ElasticNet.fit(x, y, alpha: 0.05, sample_weights: weights) end,
          [x, y, weights]
        )

      expected = ElasticNet.fit(x, y, alpha: 0.05, sample_weights: weights)

      assert_all_close(model.coefficients, expected.coefficients, atol: 1.0e-6, rtol: 1.0e-6)
      assert_all_close(model.intercept, expected.intercept, atol: 1.0e-6, rtol: 1.0e-6)
    end
  end

  describe "predict" do
    test "matches scikit-learn" do
      {x, y} = diabetes_data()

      model = ElasticNet.fit(x, y, alpha: 0.5, l1_ratio: 0.9)

      expected = Nx.tensor([205.14453125, 99.29103088378906, 173.23440551757812])
      assert_all_close(ElasticNet.predict(model, x[0..2]), expected, atol: 1.0e-2, rtol: 1.0e-5)
    end
  end

  describe "errors" do
    test "invalid :l1_ratio" do
      {x, y} = small_data()

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :l1_ratio option: expected :l1_ratio to be a number in the range [0, 1], got: 1.5",
                   fn -> ElasticNet.fit(x, y, l1_ratio: 1.5) end
    end

    test "negative :alpha" do
      {x, y} = small_data()

      assert_raise NimbleOptions.ValidationError, fn -> ElasticNet.fit(x, y, alpha: -1.0) end
    end

    test "targets with more than one column" do
      {x, y} = small_data()
      y = Nx.stack([y, y], axis: 1)

      assert_raise ArgumentError,
                   "Scholar.Linear.ElasticNet expected y to have shape {n_samples}, got tensor with shape: {5, 2}",
                   fn -> ElasticNet.fit(x, y) end
    end
  end
end
