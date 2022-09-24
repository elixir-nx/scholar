defmodule Scholar.Decomposition.PCATest do
  use ExUnit.Case, async: true

  @x Nx.tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

  test "fit test - all default options" do
    model = Scholar.Decomposition.PCA.fit(@x)
    assert model.components == Nx.tensor([[-0.83849224, -0.54491354], [0.54491354, -0.83849224]])
    assert model.explained_variance == Nx.tensor([7.9395432472229, 0.060456883162260056])
    assert model.explained_variance_ratio == Nx.tensor([0.9924429059028625, 0.007557110395282507])
    assert model.singular_values == Nx.tensor([6.30061232, 0.54980396])
    assert model.num_components == Nx.tensor(2)
    assert model.num_samples == Nx.tensor(6)
    assert model.num_features == Nx.tensor(2)
  end

  test "fit test - :num_components is float" do
    model = Scholar.Decomposition.PCA.fit(@x, num_components: 0.8)
    assert model.num_components == Nx.tensor(1)
  end

  test "fit test - :num_components is integer" do
    model = Scholar.Decomposition.PCA.fit(@x, num_components: 1)
    assert model.num_components == Nx.tensor(1)
  end

  test "transform test - :whiten set to false" do
    model = Scholar.Decomposition.PCA.fit(@x)
    num_components = model.num_components |> Nx.to_number()

    assert Scholar.Decomposition.PCA.transform(model, num_components: num_components) ==
             Nx.tensor([
               [1.3834058046340942, 0.2935786843299866],
               [2.221898078918457, -0.2513348460197449],
               [3.605304002761841, 0.04224385693669319],
               [-1.3834058046340942, -0.2935786843299866],
               [-2.221898078918457, 0.2513348460197449],
               [-3.605304002761841, -0.04224385693669319]
             ])
  end

  test "transform test - :whiten set to true" do
    model = Scholar.Decomposition.PCA.fit(@x)
    num_components = model.num_components |> Nx.to_number()

    assert Scholar.Decomposition.PCA.transform(model, num_components: num_components, whiten: true) ==
             Nx.tensor([
               [0.49096646904945374, 1.1939927339553833],
               [0.7885448336601257, -1.0221858024597168],
               [1.2795113325119019, 0.1718069314956665],
               [-0.49096646904945374, -1.1939927339553833],
               [-0.7885448336601257, 1.0221858024597168],
               [-1.2795113325119019, -0.1718069314956665]
             ])
  end

  describe "errors" do
    test "input rank different than 2" do
      assert_raise ArgumentError,
                   "expected input to have shape {n_samples, n_features}, got tensor with shape: {4}",
                   fn ->
                     Scholar.Decomposition.PCA.fit(Nx.tensor([1, 2, 3, 4]))
                   end
    end

    test "fit test - :num_components is atom" do
      assert_raise ArgumentError,
                   ":num_components must be :none, a number, or a tensor, got: :two",
                   fn ->
                     Scholar.Decomposition.PCA.fit(@x, num_components: :two)
                   end
    end

    test "transform test - missing :num_components" do
      assert_raise NimbleOptions.ValidationError,
                   "required :num_components option not found, received options: []",
                   fn ->
                     model = Scholar.Decomposition.PCA.fit(@x)
                     Scholar.Decomposition.PCA.transform(model)
                   end
    end

    test "transform test - :whiten is not boolean" do
      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :whiten option: expected boolean, got: :yes",
                   fn ->
                     model = Scholar.Decomposition.PCA.fit(@x)
                     num_components = model.num_components |> Nx.to_number()

                     Scholar.Decomposition.PCA.transform(model,
                       num_components: num_components,
                       whiten: :yes
                     )
                   end
    end
  end
end
