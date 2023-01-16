defmodule Scholar.PreprocessingTest do
  use Scholar.Case, async: true
  alias Scholar.Preprocessing
  doctest Preprocessing

  describe "standard_scaler/1" do
    test "applies standard scaling to data" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])

      expected =
        Nx.tensor([
          [0.5212860703468323, -1.3553436994552612, 1.4596009254455566],
          [1.4596009254455566, -0.4170288145542145, -0.4170288145542145],
          [-0.4170288145542145, 0.5212860703468323, -1.3553436994552612]
        ])

      assert_all_close(Preprocessing.standard_scale(data), expected)
    end

    test "leaves data as it is when variance is zero" do
      data = 42.0
      expected = Nx.tensor(data)
      assert Preprocessing.standard_scale(data) == expected
    end
  end

  describe "polynomial feature matrix" do
    test "transform/1 degree=2 fit_intercept?=false returns the input" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])

      assert Preprocessing.polynomial_transform(data, degree: 1, fit_intercept?: false) == data
    end

    test "transform/1 degree=2 only adds intercept" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      expected = Nx.tensor([[1, 1, -1, 2], [1, 2, 0, 0], [1, 0, 1, -1]])

      assert Preprocessing.polynomial_transform(data, degree: 1) == expected
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

      assert Preprocessing.polynomial_transform(data) == expected
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

      assert Preprocessing.polynomial_transform(data, degree: 4, fit_intercept?: false) ==
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

      assert Preprocessing.polynomial_transform(data, degree: 3) == expected
    end
  end
end
