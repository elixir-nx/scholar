defmodule Scholar.Preprocessing.ScalerTest do
  use ExUnit.Case, async: true

  alias Scholar.Preprocessing.Scaler

  describe "fit/1" do
    test "should return a scaler structure with argument is a list" do
      data = Nx.tensor([1, 2, 3])
      assert %Scaler{data: ^data, transformation: nil} = Scaler.fit([1, 2, 3])
    end

    test "should return a scaler structure with argument is a tensor" do
      data = Nx.tensor([1, 2, 3])
      assert %Scaler{data: ^data, transformation: nil} = Scaler.fit(data)
    end

    test "should return an empty list" do
      assert [] = Scaler.fit([])
    end

    test "should return the argument back" do
      assert [1] = Scaler.fit([1])
    end

    test "should raise an exception" do
      assert_raise ArgumentError, fn ->
        Scaler.fit(42)
      end
    end
  end

  describe "transform/1" do
    test "should transform the data" do
      data = Nx.tensor([1, 2, 3])
      transformation = Nx.tensor([-1.2247447967529297, 0.0, 1.2247447967529297])
      scaler = Scaler.fit(data)
      assert %Scaler{data: ^data, transformation: ^transformation} = Scaler.transform(scaler)
    end

    test "should raise an exception" do
      assert_raise ArgumentError, fn ->
        Scaler.transform(42)
      end
    end
  end
end
