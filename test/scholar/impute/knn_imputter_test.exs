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

      # row 3 has three missing values; nearest donors per sklearn are rows 1 and 4, not 0 and 1
      assert statistics ==
               Nx.tensor([
                 [:nan, :nan, :nan, :nan],
                 [:nan, :nan, :nan, :nan],
                 [:nan, :nan, 4.0, 5.0],
                 [10.0, 11.0, 6.0, :nan],
                 [:nan, :nan, :nan, :nan]
               ])

      assert jit_transform.(knn_imputer, x) ==
               Nx.tensor([
                 [0.0, 1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0, 7.0],
                 [8.0, 9.0, 4.0, 5.0],
                 [10.0, 11.0, 6.0, 15.0],
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

      # with 1 neighbor, sklearn picks row 4 as row 3's closest donor, not row 0
      assert statistics ==
               Nx.tensor([
                 [:nan, :nan, :nan, :nan],
                 [:nan, :nan, :nan, :nan],
                 [:nan, :nan, 6.0, 7.0],
                 [16.0, 17.0, 6.0, :nan],
                 [:nan, :nan, :nan, :nan]
               ])

      assert jit_transform.(knn_imputter, x) ==
               Nx.tensor([
                 [0.0, 1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0, 7.0],
                 [8.0, 9.0, 6.0, 7.0],
                 [16.0, 17.0, 6.0, 15.0],
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

    test "picks donors by actual distance for a row with two missing values, not by row index" do
      # rows 0/1 are low-index but far from row 2; rows 5/6 are high-index but close.
      # sklearn.impute.KNNImputer(n_neighbors=2) imputes row 2 to [55.0, 85.0, 5.0]
      x =
        Nx.tensor([
          [1.0, 2.0, 500.0],
          [1.1, 2.1, 600.0],
          [:nan, :nan, 5.0],
          [1.3, 2.3, :nan],
          [10.0, 20.0, 700.0],
          [50.0, 80.0, 5.1],
          [60.0, 90.0, 4.9]
        ])

      imputer = KNNImputter.fit(x, num_neighbors: 2)
      result = KNNImputter.transform(imputer, x)

      assert_all_close(result[2], Nx.tensor([55.0, 85.0, 5.0]))
      # row 3 has a single missing value, unaffected by the bug this fixes
      assert_all_close(result[3], Nx.tensor([1.3, 2.3, 550.0]))
    end

    test "keeps the input's float type through fit and transform" do
      # the distance accumulator and NaN placeholder used to hardcode f32
      x =
        Nx.tensor(
          [
            [0.1, 0.2, 500.123456789],
            [0.11, 0.21, 600.987654321],
            [:nan, :nan, 5.000000001],
            [50.0, 80.0, 5.100000002],
            [60.0, 90.0, 4.900000003]
          ],
          type: :f64
        )

      imputer = KNNImputter.fit(x, num_neighbors: 2)
      result = KNNImputter.transform(imputer, x)

      assert Nx.type(result) == {:f, 64}
      assert_all_close(result[2], Nx.tensor([55.0, 85.0, 5.0], type: :f64), atol: 1.0e-9)
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
