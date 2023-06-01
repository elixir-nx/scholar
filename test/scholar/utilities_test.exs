defmodule Scholar.UtilitiesTest do
  use Scholar.Case, async: true
  alias Scholar.Utilities
  # doctest Utilities

  describe "train_test_split/2" do
    test "Split into 50% for training and 50% for testing" do
      tensor = Nx.iota({10, 2}, names: [:x, :y])
      {train, test} = Utilities.train_test_split(tensor, train_size: 0.5)

      assert Nx.tensor([
               [0, 1],
               [2, 3],
               [4, 5],
               [6, 7],
               [8, 9]
             ]) == train

      assert Nx.tensor([
               [10, 11],
               [12, 13],
               [14, 15],
               [16, 17],
               [18, 19]
             ]) == test
    end

    test "Split into 70% for training and 30% for testing" do
      tensor = Nx.iota({100, 6}, names: [:x, :y])
      {train, test} = Utilities.train_test_split(tensor, train_size: 0.7)

      assert length(Nx.to_list(train)) == 70
      assert length(Nx.to_list(test)) == 30
    end

    test "Split into 75% for training and 25% for testing" do
      tensor = Nx.iota({100, 10}, names: [:x, :y])
      {train, test} = Utilities.train_test_split(tensor, train_size: 0.75)

      assert length(Nx.to_list(train)) == 75
      assert length(Nx.to_list(test)) == 25
    end

    test "Split into 61% for training and 39% for testing" do
      tensor = Nx.iota({100, 10}, names: [:x, :y])
      {train, test} = Utilities.train_test_split(tensor, train_size: 0.61)

      assert length(Nx.to_list(train)) == 61
      assert length(Nx.to_list(test)) == 39
    end

    test "Split into 60% for training and 40% for testing with unbalanced data" do
      tensor = Nx.iota({73, 4}, names: [:x, :y])
      {train, test} = Utilities.train_test_split(tensor, train_size: 0.61)

      assert length(Nx.to_list(train)) == 44
      assert length(Nx.to_list(test)) == 29
    end
  end
end
