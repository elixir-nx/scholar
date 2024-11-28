defmodule KNNImputterTest do
  use Scholar.Case, async: true
  alias Scholar.Impute.KNNImputter
  doctest KNNImputter

  describe "general cases" do
    def generate_data() do
      x = Nx.iota({5, 4})
      x = Nx.select(Nx.equal(Nx.quotient(x, 5), 2), Nx.Constants.nan(), x)
      Nx.indexed_put(x, Nx.tensor([[4, 2]]), Nx.tensor([6.0]))
    end

    test "general KNN imputer" do
      x = generate_data()
      jit_fit = Nx.Defn.jit(&KNNImputter.fit/2)
      jit_transform = Nx.Defn.jit(&KNNImputter.transform/2)

      knn_imputer =
        %KNNImputter{statistics: statistics, missing_values: missing_values} =
        jit_fit.(x, missing_values: :nan, num_neighbors: 2)

      assert missing_values == :nan

      assert statistics ==
               Nx.tensor([
                 [:nan, :nan, :nan, :nan],
                 [:nan, :nan, :nan, :nan],
                 [:nan, :nan, 4.0, 5.0],
                 [2.0, 3.0, 4.0, :nan],
                 [:nan, :nan, :nan, :nan]
               ])

      assert jit_transform.(knn_imputer, x) ==
               Nx.tensor([
                 [0.0, 1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0, 7.0],
                 [8.0, 9.0, 4.0, 5.0],
                 [2.0, 3.0, 4.0, 15.0],
                 [16.0, 17.0, 6.0, 19.0]
               ])
    end

    test "general KNN imputer with different number of neighbors" do
      x = generate_data()
      jit_fit = Nx.Defn.jit(&KNNImputter.fit/2)
      jit_transform = Nx.Defn.jit(&KNNImputter.transform/2)

      knn_imputter =
        %KNNImputter{statistics: statistics, missing_values: missing_values} =
        jit_fit.(x, missing_values: :nan, num_neighbors: 1)

      assert missing_values == :nan

      assert statistics ==
               Nx.tensor([
                 [:nan, :nan, :nan, :nan],
                 [:nan, :nan, :nan, :nan],
                 [:nan, :nan, 2.0, 3.0],
                 [0.0, 1.0, 2.0, :nan],
                 [:nan, :nan, :nan, :nan]
               ])

      assert jit_transform.(knn_imputter, x) ==
               Nx.tensor([
                 [0.0, 1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0, 7.0],
                 [8.0, 9.0, 2.0, 3.0],
                 [0.0, 1.0, 2.0, 15.0],
                 [16.0, 17.0, 6.0, 19.0]
               ])
    end

    test "missing values different than :nan" do
      x = generate_data()
      x = Nx.select(Nx.is_nan(x), 19.0, x)
#      x = Nx.select(Nx.equal(x,19), :nan, x)
      jit_fit = Nx.Defn.jit(&KNNImputter.fit/2)
      jit_transform = Nx.Defn.jit(&KNNImputter.transform/2)

      knn_imputter =
        %KNNImputter{statistics: statistics, missing_values: missing_values} =
        jit_fit.(x, missing_values: 19.0, num_neighbors: 2)

      assert missing_values == 19.0

      assert statistics ==
               Nx.tensor([
                 [:nan, :nan, :nan, :nan],
                 [:nan, :nan, :nan, :nan],
                 [:nan, :nan, 4.0, 5.0],
                 [2.0, 3.0, 4.0, :nan],
                 [:nan, :nan, :nan, 5.0]
               ])

      assert jit_transform.(knn_imputter, x) ==
               Nx.tensor([
                 [0.0, 1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0, 7.0],
                 [8.0, 9.0, 4.0, 5.0],
                 [2.0, 3.0, 4.0, 15.0],
                 [16.0, 17.0, 6.0, 5.0]
               ])
    end
  end

  describe "errors" do
    test "invalid impute rank" do
      x = Nx.tensor([1, 2, 2, 3])

      assert_raise ArgumentError,
                   "wrong input rank. Expected: 2, got: 1",
                   fn ->
                     KNNImputter.fit(x, missing_values: 1, num_neighbors: 2)
                   end
    end

    test "invalid n_neighbors value" do
      x = generate_data()

      jit_fit = Nx.Defn.jit(&KNNImputter.fit/2)

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :num_neighbors option: expected positive integer, got: -1",
                   fn ->
                     jit_fit.(x, missing_values: 1.0, num_neighbors: -1)
                   end
    end
  end
end
