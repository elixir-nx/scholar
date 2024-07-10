defmodule Scholar.Linear.PolynomialRegressionTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.PolynomialRegression
  doctest PolynomialRegression

  describe "fit" do
    test "matches sklearn for shapes {4, 6}, {4}; degree 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [
            0.9106853604316711,
            0.2549760937690735,
            0.6961511969566345,
            0.9727275371551514,
            0.21542717516422272,
            0.1134142205119133
          ],
          [
            0.06984847038984299,
            0.5999580025672913,
            0.5002026557922363,
            0.3873320519924164,
            0.4304788112640381,
            0.7481111884117126
          ],
          [
            0.8257377743721008,
            0.3594258427619934,
            0.08661065995693207,
            0.5118331909179688,
            0.38879409432411194,
            0.3640798032283783
          ],
          [
            0.060781270265579224,
            0.054385241121053696,
            0.6188914775848389,
            0.9003549218177795,
            0.7764868140220642,
            0.9584161043167114
          ]
        ])

      b =
        Nx.tensor([0.608439028263092, 0.6562057137489319, 0.9454836249351501, 0.8614323735237122])

      expected_coeff =
        Nx.tensor(
          [0.07950567, -0.08133664, -0.15176095, -0.00444476, 0.07165004] ++
            [0.02493282, 0.05871655, 0.03970227, -0.10034085, -0.03266322] ++
            [0.06277008, 0.069476, -0.06337871, -0.09605422, -0.04177537] ++
            [-0.02704311, -0.07255626, -0.10330329, -0.09088593, -0.01088266] ++
            [-0.02450841, -0.02890116, 0.07061561, 0.07474134, 0.06896995] ++
            [0.05317045, 0.01400374]
        )

      expected_intercept = Nx.tensor(0.8032849746598018)

      %PolynomialRegression{
        coefficients: actual_coeff,
        intercept: actual_intercept
      } = PolynomialRegression.fit(a, b, degree: 2)

      assert_all_close(expected_coeff, actual_coeff, rtol: 1.0e-2, atol: 1.0e-2)
      assert_all_close(expected_intercept, actual_intercept, rtol: 1.0e-2, atol: 1.0e-2)
    end
  end

  def a do
    Nx.tensor([
      [
        0.7731901407241821,
        0.5813425779342651,
        0.8365984559059143,
        0.2182593196630478,
        0.06448899209499359,
        0.9420031905174255
      ],
      [
        0.6547101736068726,
        0.05023770406842232,
        0.657528281211853,
        0.24924135208129883,
        0.8238568902015686,
        0.11182288080453873
      ],
      [
        0.7693489193916321,
        0.6696648001670837,
        0.6877049803733826,
        0.08740159869194031,
        0.6053816676139832,
        0.5419610142707825
      ],
      [
        0.03419172018766403,
        0.8298202753067017,
        0.6097439527511597,
        0.0184243805706501,
        0.5578944087028503,
        0.9986271858215332
      ]
    ])
  end

  def b do
    Nx.tensor([
      0.38682249188423157,
      0.8040792346000671,
      0.8069542646408081,
      0.3620224595069885
    ])
  end

  describe "predict" do
    test "predict when :fit_intercept? set to true, degree set to 2" do
      a = a()

      b = b()

      sample_weights = [
        0.8669093251228333,
        0.10421276837587357,
        0.996828556060791,
        0.29747673869132996
      ]

      prediction_input = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
      expected_prediction = Nx.tensor([-3.61666677])

      actual_prediction =
        PolynomialRegression.fit(a, b,
          degree: 2,
          sample_weights: sample_weights,
          fit_intercept?: true
        )
        |> PolynomialRegression.predict(prediction_input)

      assert_all_close(expected_prediction, actual_prediction, rtol: 1.0e-2, atol: 1.0e-2)
    end

    test "predict when :fit_intercept? set to false, degree set to 3" do
      a = a()
      b = b()

      sample_weights = [
        0.8669093251228333,
        0.10421276837587357,
        0.996828556060791,
        0.29747673869132996
      ]

      prediction_input =
        Nx.tensor([
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ])

      expected_prediction = Nx.tensor([37.31348465, 37.31348465])

      actual_prediction =
        PolynomialRegression.fit(a, b,
          degree: 3,
          sample_weights: sample_weights,
          fit_intercept?: false
        )
        |> PolynomialRegression.predict(prediction_input)

      assert_all_close(expected_prediction, actual_prediction, rtol: 1.0e-2, atol: 1.0e-2)
    end
  end

  describe "polynomial feature matrix" do
    test "transform/1 degree=2 fit_intercept?=false returns the input" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])

      assert PolynomialRegression.transform(data, degree: 1, fit_intercept?: false) ==
               data
    end

    test "transform/1 degree=1 only adds intercept" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      expected = Nx.tensor([[1, 1, -1, 2], [1, 2, 0, 0], [1, 0, 1, -1]])

      assert PolynomialRegression.transform(data, degree: 1) == expected
    end

    test "transform/1 degree=2" do
      data = Nx.iota({3, 2})
      # Results compared against Scipy
      expected =
        Nx.tensor([
          [1, 0, 1, 0, 0, 1],
          [1, 2, 3, 4, 6, 9],
          [1, 4, 5, 16, 20, 25]
        ])

      assert PolynomialRegression.transform(data) == expected
    end

    test "transform/1 degree=4 fit_intercept?=false" do
      data = Nx.iota({3, 2})
      # Results compared against Scipy
      expected =
        Nx.tensor([
          [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
          [2, 3, 4, 6, 9, 8, 12, 18, 27, 16, 24, 36, 54, 81],
          [4, 5, 16, 20, 25, 64, 80, 100, 125, 256, 320, 400, 500, 625]
        ])

      assert PolynomialRegression.transform(data, degree: 4, fit_intercept?: false) ==
               expected
    end

    test "transform/1 degree=3" do
      data = Nx.iota({1, 5})

      expected =
        Nx.tensor([
          [1, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 6, 8, 9, 12] ++
            [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3] ++
            [4, 4, 6, 8, 9, 12, 16, 8, 12, 16, 18, 24, 32, 27, 36, 48, 64]
        ])

      assert PolynomialRegression.transform(data, degree: 3) == expected
    end

    test "transform/1 degree=3 multiple samples (multiple features)" do
      data = Nx.tensor([[2, 3, 5], [0, 1, 2]])

      expected =
        Nx.tensor([
          [1, 2, 3, 5, 4, 6, 10, 9, 15, 25, 8, 12, 20, 18, 30, 50, 27, 45, 75, 125],
          [1, 0, 1, 2, 0, 0, 0, 1, 2, 4, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8]
        ])

      assert PolynomialRegression.transform(data, degree: 3) == expected
    end

    test "transform/1 degree=3 fit_intercept?=false (high number of samples)" do
      data = Nx.tensor([[2, 3, 5, 7]])

      expected =
        Nx.tensor([
          [2, 3, 5, 7, 4, 6, 10, 14, 9, 15, 21, 25, 35, 49, 8, 12, 20, 28, 18] ++
            [30, 42, 50, 70, 98, 27, 45, 63, 75, 105, 147, 125, 175, 245, 343]
        ])

      assert PolynomialRegression.transform(data, degree: 3, fit_intercept?: false) ==
               expected
    end

    test "transform/1 degree=6 fit_intercept?=false (high degree)" do
      data = Nx.tensor([[2, 3]])

      expected =
        Nx.tensor([
          [2, 3, 4, 6, 9, 8, 12, 18, 27, 16, 24, 36, 54, 81, 32] ++
            [48, 72, 108, 162, 243, 64, 96, 144, 216, 324, 486, 729]
        ])

      assert PolynomialRegression.transform(data, degree: 6, fit_intercept?: false) ==
               expected
    end
  end

  describe "column target tests" do
    test "fit column target" do
      a = a()
      b = b()

      sample_weights = [
        0.8669093251228333,
        0.10421276837587357,
        0.996828556060791,
        0.29747673869132996
      ]

      prediction_input = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

      model =
        PolynomialRegression.fit(a, b,
          degree: 2,
          sample_weights: sample_weights,
          fit_intercept?: true
        )

      prediction = PolynomialRegression.predict(model, prediction_input)

      col_model =
        PolynomialRegression.fit(a, b |> Nx.new_axis(-1),
          degree: 2,
          sample_weights: sample_weights,
          fit_intercept?: true
        )

      col_prediction = PolynomialRegression.predict(col_model, prediction_input)
      assert model == col_model
      assert prediction == col_prediction
    end
  end
end
