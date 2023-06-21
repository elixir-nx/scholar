defmodule SimpleImputerTest do
  use Scholar.Case, async: true
  alias Scholar.Impute.SimpleImputer
  doctest SimpleImputer

  describe "general cases" do
    def generate_data() do
      x = Nx.iota({5, 4})
      x = Nx.select(Nx.equal(Nx.quotient(x, 5), 2), Nx.Constants.nan(), x)
      Nx.indexed_put(x, Nx.tensor([[4, 2]]), Nx.tensor([6.0]))
    end

    test "general test mode" do
      x = generate_data()

      simple_imputer_mode =
        %SimpleImputer{statistics: statistics, missing_values: missing_values} =
        SimpleImputer.fit(x, missing_values: :nan, strategy: :mode)

      assert statistics == Nx.tensor([0.0, 1.0, 6.0, 3.0])
      assert missing_values == :nan

      assert SimpleImputer.transform(simple_imputer_mode, x) ==
               Nx.tensor([
                 [0.0, 1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0, 7.0],
                 [8.0, 9.0, 6.0, 3.0],
                 [0.0, 1.0, 6.0, 15.0],
                 [16.0, 17.0, 6.0, 19.0]
               ])
    end

    test "general test median" do
      x = generate_data()

      simple_imputer_median =
        %SimpleImputer{statistics: statistics, missing_values: missing_values} =
        SimpleImputer.fit(x, missing_values: :nan, strategy: :median)

      assert statistics == Nx.tensor([6.0, 7.0, 6.0, 11.0])
      assert missing_values == :nan

      assert SimpleImputer.transform(simple_imputer_median, x) ==
               Nx.tensor([
                 [0.0, 1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0, 7.0],
                 [8.0, 9.0, 6.0, 11.0],
                 [6.0, 7.0, 6.0, 15.0],
                 [16.0, 17.0, 6.0, 19.0]
               ])
    end

    test "general test mean" do
      x = generate_data()

      simple_imputer_mean =
        %SimpleImputer{statistics: statistics, missing_values: missing_values} =
        SimpleImputer.fit(x, missing_values: :nan, strategy: :mean)

      assert_all_close(statistics , Nx.tensor([7.0, 8.0, 4.666666507720947, 11.0]))
      assert missing_values == :nan

      assert_all_close(
        SimpleImputer.transform(simple_imputer_mean, x),
        Nx.tensor([
          [0.0, 1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0, 7.0],
          [8.0, 9.0, 4.666666507720947, 11.0],
          [7.0, 8.0, 4.666666507720947, 15.0],
          [16.0, 17.0, 6.0, 19.0]
        ])
      )
    end

    test "general test constant value" do
      x = generate_data()

      simple_imputer_constant_with_zeros =
        %SimpleImputer{statistics: statistics, missing_values: missing_values} =
        SimpleImputer.fit(x, missing_values: :nan, strategy: :constant)

      assert statistics == Nx.tensor([0.0, 0.0, 0.0, 0.0])
      assert missing_values == :nan

      %SimpleImputer{statistics: statistics, missing_values: missing_values} =
        SimpleImputer.fit(x,
          missing_values: :nan,
          strategy: :constant,
          fill_value: 1.37
        )

      assert statistics == Nx.tensor([1.37, 1.37, 1.37, 1.37])
      assert missing_values == :nan

      assert SimpleImputer.transform(simple_imputer_constant_with_zeros, x) ==
               Nx.tensor([
                 [0.0, 1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0, 7.0],
                 [8.0, 9.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 15.0],
                 [16.0, 17.0, 6.0, 19.0]
               ])
    end
  end

  test "mode with integer type" do
    x = Nx.tile(Nx.tensor([1, 2, 1, 2, 1, 2]), [5, 1]) |> Nx.reshape({6, 5})

    simple_imputer_constant_with_zeros =
      %SimpleImputer{statistics: statistics, missing_values: missing_values} =
      SimpleImputer.fit(x, missing_values: 1, strategy: :mode)

    assert statistics == Nx.tensor([2, 2, 2, 2, 2])
    assert missing_values == 1

    assert SimpleImputer.transform(simple_imputer_constant_with_zeros, x) ==
             Nx.broadcast(2, {6, 5})
  end

  describe "errors" do
    test "Wrong impute rank" do
      x = Nx.tensor([1, 2, 2, 3])

      assert_raise ArgumentError,
                   "Wrong input rank. Expected: 2, got: 1",
                   fn ->
                     SimpleImputer.fit(x, missing_values: 1, strategy: :mode)
                   end
    end

    test "Collision of nan" do
      x = generate_data()

      assert_raise ArgumentError,
                   ":missing_values other than :nan possible only if there is no Nx.Constant.nan() in the array",
                   fn ->
                     SimpleImputer.fit(x, missing_values: 1.0, strategy: :mode)
                   end
    end

    test "Wrong :fill_value type" do
      x = Nx.tensor([[1.0, 2.0, 2.0, 3.0]])

      assert_raise ArgumentError,
                   "Wrong type of `:fill_value` for the given data. Expected: :f or :bf, got: :s",
                   fn ->
                     SimpleImputer.fit(x,
                       missing_values: 1.0,
                       strategy: :constant,
                       fill_value: 2
                     )
                   end
    end
  end
end
