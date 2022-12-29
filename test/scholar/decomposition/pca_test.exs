defmodule Scholar.Decomposition.PCATest do
  use ExUnit.Case, async: true
  import ScholarCase


  @x Nx.tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
  @x2 Nx.tensor([[1, 4], [54, 6], [26, 7]])
  @x3 Nx.tensor([[-1, -1, 3], [-2, -1, 2], [-3, -2, 1], [3, 1, 1], [21, 2, 1], [5, 3, 2]])

  test "fit test - all default options" do
    model = Scholar.Decomposition.PCA.fit(@x)

    assert_all_close(
      model.components,
      Nx.tensor([[-0.83849224, -0.54491354], [0.54491354, -0.83849224]]),
      atol: 1.0e-3
    )

    assert_all_close(model.explained_variance, Nx.tensor([7.9395432472229, 0.060456883162260056]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.explained_variance_ratio,
      Nx.tensor([0.9924429059028625, 0.007557110395282507])
    )

    assert_all_close(model.singular_values, Nx.tensor([6.30061232, 0.54980396]), atol: 1.0e-3)
    assert model.mean == Nx.tensor([0.0, 0.0])
    assert model.num_components == 2
    assert model.num_samples == Nx.tensor(6)
    assert model.num_features == Nx.tensor(2)
  end

  test "fit test - :num_components is integer" do
    model = Scholar.Decomposition.PCA.fit(@x, num_components: 1)
    assert model.num_components == 1
  end

  test "transform test - :whiten set to false" do
    model = Scholar.Decomposition.PCA.fit(@x)

    assert_all_close(
      Scholar.Decomposition.PCA.transform(model, @x),
      Nx.tensor([
        [1.3834056854248047, 0.2935786843299866],
        [2.221898078918457, -0.2513348460197449],
        [3.6053037643432617, 0.0422438383102417],
        [-1.3834056854248047, -0.2935786843299866],
        [-2.221898078918457, 0.2513348460197449],
        [-3.6053037643432617, -0.0422438383102417]
      ]),
      atol: 1.0e-2
    )
  end

  test "transform test - :whiten set to false and and num components different than min(num_samples, num_components)" do
    model = Scholar.Decomposition.PCA.fit(@x3, num_components: 2)

    assert_all_close(
      model.components,
      Nx.tensor([
        [0.98732591, 0.15474766, -0.03522361],
        [-0.14912572, 0.98053261, 0.12773922085762024]
      ]),
      atol: 1.0e-2
    )

    assert_all_close(
      model.explained_variance,
      Nx.tensor([82.19153594970703, 1.966333031654358]),
      atol: 1.0e-2
    )

    assert_all_close(
      model.explained_variance_ratio,
      Nx.tensor([0.97038421, 0.023215265944600105]),
      atol: 1.0e-2
    )

    assert_all_close(model.singular_values, Nx.tensor([20.272090911865234, 3.13554849]),
      atol: 1.0e-2
    )

    assert_all_close(model.mean, Nx.tensor([3.83333333, 0.33333333, 1.66666667]), atol: 1.0e-2)
    assert model.num_components == 2
    assert model.num_samples == Nx.tensor(6)
    assert model.num_features == Nx.tensor(3)

    assert_all_close(
      Scholar.Decomposition.PCA.transform(model, @x3),
      Nx.tensor([
        [-5.02537027, -0.41628357768058777],
        [-5.977472305297852, -0.39489707350730896],
        [-7.084321975708008, -1.3540430068969727],
        [-0.6961240172386169, 0.6928002834320068],
        [17.23049002, -1.010930061340332],
        [1.5527995824813843, 2.48335338]
      ]),
      atol: 1.0e-2
    )
  end

  test "transform test - :whiten set to false and different data in fit and transform" do
    model = Scholar.Decomposition.PCA.fit(@x, num_components: 2)

    assert_all_close(
      Scholar.Decomposition.PCA.transform(model, @x2),
      Nx.tensor([
        [-3.018146276473999, -2.8090553283691406],
        [-48.54806137084961, 24.394376754760742],
        [-25.615192413330078, 8.298306465148926]
      ]),
      atol: 1.0e-1,
      rtol: 1.0e-3
    )
  end

  test "transform test - :whiten set to true" do
    model = Scholar.Decomposition.PCA.fit(@x)

    assert_all_close(
      Scholar.Decomposition.PCA.transform(model, @x, whiten: true),
      Nx.tensor([
        [0.49096643924713135, 1.1939926147460938],
        [0.7885448336601257, -1.0221858024597168],
        [1.2795112133026123, 0.17180685698986053],
        [-0.49096643924713135, -1.1939926147460938],
        [-0.7885448336601257, 1.0221858024597168],
        [-1.2795112133026123, -0.17180685698986053]
      ]),
      atol: 1.0e-2
    )
  end

  test "fit_transform test - :whiten set to false" do
    model = Scholar.Decomposition.PCA.fit(@x)

    assert_all_close(
      Scholar.Decomposition.PCA.transform(model, @x),
      Scholar.Decomposition.PCA.fit_transform(@x),
      atol: 1.0e-2
    )
  end

  test "fit_transform test - :whiten set to false and and num components different than min(num_samples, num_components)" do
    model = Scholar.Decomposition.PCA.fit(@x3, num_components: 2)

    assert_all_close(
      Scholar.Decomposition.PCA.transform(model, @x3),
      Scholar.Decomposition.PCA.fit_transform(@x3, num_components: 2),
      atol: 1.0e-2
    )
  end

  test "fit_transform test - :whiten set to true" do
    model = Scholar.Decomposition.PCA.fit(@x)

    assert_all_close(
      Scholar.Decomposition.PCA.transform(model, @x, whiten: true),
      Scholar.Decomposition.PCA.fit_transform(@x, whiten: true),
      atol: 1.0e-2
    )
  end

  describe "errors" do
    test "input rank different than 2" do
      assert_raise ArgumentError,
                   "expected x to have rank equal to: 2, got: 1",
                   fn ->
                     Scholar.Decomposition.PCA.fit(Nx.tensor([1, 2, 3, 4]))
                   end
    end

    test "fit test - :num_components bigger than min(num_samples, num_features)" do
      assert_raise ArgumentError,
                   "expected :num_components to be integer in range 1 to 2, got: 4",
                   fn ->
                     Scholar.Decomposition.PCA.fit(@x, num_components: 4)
                   end
    end

    test "fit test - :num_components is atom" do
      assert_raise NimbleOptions.ValidationError,
                   """
                   expected :num_components option to match at least one given type, but didn't match any. Here are the reasons why it didn't match each of the allowed types:

                     * invalid value for :num_components option: expected one of [nil], got: :two
                     * invalid value for :num_components option: expected positive integer, got: :two\
                   """,
                   fn ->
                     Scholar.Decomposition.PCA.fit(@x, num_components: :two)
                   end
    end

    test "transform test - :whiten is not boolean" do
      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :whiten option: expected boolean, got: :yes",
                   fn ->
                     model = Scholar.Decomposition.PCA.fit(@x)

                     Scholar.Decomposition.PCA.transform(model, @x, whiten: :yes)
                   end
    end
  end
end
