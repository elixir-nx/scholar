defmodule Scholar.PolynomialTest do
  use Scholar.Case, async: true
  alias Scholar.Polynomial
  doctest Polynomial

  describe "polynomial feature matrix" do
    test "transform/1 degree=2 fit_intercept?=false returns the input" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])

      assert Polynomial.transform(data, degree: 1, fit_intercept?: false) == data
    end

    test "transform/1 degree=2 only adds intercept" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      expected = Nx.tensor([[1, 1, -1, 2], [1, 2, 0, 0], [1, 0, 1, -1]])

      assert Polynomial.transform(data, degree: 1) == expected
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

      assert Polynomial.transform(data) == expected
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

      assert Polynomial.transform(data, degree: 4, fit_intercept?: false) ==
               expected
    end

    test "" do
      data = Nx.iota({1, 5})

      expected =
        Nx.tensor([
          [
            1,
            0,
            1,
            2,
            3,
            4,
            0,
            0,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            4,
            6,
            8,
            9,
            12,
            16,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            4,
            6,
            8,
            9,
            12,
            16,
            8,
            12,
            16,
            18,
            24,
            32,
            27,
            36,
            48,
            64
          ]
        ])

      assert Polynomial.transform(data, degree: 3) == expected
    end

    test "transform/1 degree=3 multiple samples (multiple features)" do
      data = Nx.tensor([[2, 3, 5], [0, 1, 2]])

      expected =
        Nx.tensor([
          [1, 2, 3, 5, 4, 6, 10, 9, 15, 25, 8, 12, 20, 18, 30, 50, 27, 45, 75, 125],
          [1, 0, 1, 2, 0, 0, 0, 1, 2, 4, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8]
        ])

      assert Polynomial.transform(data, degree: 3) == expected
    end

    test "transform/1 degree=3 fit_intercept?=false (high number of samples)" do
      data = Nx.tensor([[2, 3, 5, 7]])

      expected =
        Nx.tensor([
          [
            2,
            3,
            5,
            7,
            4,
            6,
            10,
            14,
            9,
            15,
            21,
            25,
            35,
            49,
            8,
            12,
            20,
            28,
            18,
            30,
            42,
            50,
            70,
            98,
            27,
            45,
            63,
            75,
            105,
            147,
            125,
            175,
            245,
            343
          ]
        ])

      assert Polynomial.transform(data, degree: 3, fit_intercept?: false) == expected
    end

    test "transform/1 degree=6 fit_intercept?=false (high degree)" do
      data = Nx.tensor([[2, 3]])

      expected =
        Nx.tensor([
          [
            2,
            3,
            4,
            6,
            9,
            8,
            12,
            18,
            27,
            16,
            24,
            36,
            54,
            81,
            32,
            48,
            72,
            108,
            162,
            243,
            64,
            96,
            144,
            216,
            324,
            486,
            729
          ]
        ])

      assert Polynomial.transform(data, degree: 6, fit_intercept?: false) == expected
    end
  end
end
