defmodule Scholar.Interpolation.LinearTest do
  use ExUnit.Case, async: true

  alias Scholar.Interpolation.Linear

  describe "linear" do
    test "fit/2" do
      x = Nx.iota({4})
      y = Nx.tensor([0, 0, 1, 5])

      model = Linear.fit(x, y)

      assert model.coefficients ==
               Nx.tensor([
                 [0.0, 0.0],
                 [1.0, -1.0],
                 [4.0, -7.0]
               ])
    end

    test "input validation error cases" do
      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 1, got: {1, 1, 1}",
                   fn ->
                     Linear.fit(Nx.iota({1, 1, 1}), Nx.iota({1, 1, 1}))
                   end

      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 1, got: {}",
                   fn ->
                     Linear.fit(Nx.iota(1), Nx.iota(1))
                   end

      assert_raise ArgumentError,
                   "expected x to be a tensor with shape {n}, where n > 1, got: {1}",
                   fn ->
                     Linear.fit(Nx.iota({1}), Nx.iota({1}))
                   end

      assert_raise ArgumentError, "expected y to have shape {4}, got: {3}", fn ->
        Linear.fit(Nx.iota({4}), Nx.iota({3}))
      end
    end

    test "predict/2" do
      x = Nx.iota({4})
      y = Nx.tensor([0, 0, 1, 5])

      model = Linear.fit(x, y)

      assert Linear.predict(model, Nx.tensor([[[-0.5], [0.5], [1.5], [2.5], [3.5]]])) ==
               Nx.tensor([[[0.0], [0.0], [0.5], [3], [7]]])
    end
  end
end
