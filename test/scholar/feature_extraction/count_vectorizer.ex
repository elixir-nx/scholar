defmodule Scholar.Preprocessing.BinarizerTest do
  use Scholar.Case, async: true
  alias Scholar.FeatureExtraction.CountVectorizer
  doctest CountVectorizer

  describe "fit_transform" do
    test "fit_transform test" do
      counts = CountVectorizer.fit_transform(Nx.tensor([[2, 3, 0], [1, 4, 4]]))

      expected_counts = Nx.tensor([[1, 0, 1, 1, 0], [0, 1, 0, 0, 2]])

      assert counts == expected_counts
    end

    test "fit_transform test - tensor with padding" do
      counts = CountVectorizer.fit_transform(Nx.tensor([[2, 3, 0], [1, 4, -1]]))

      expected_counts = Nx.tensor([[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]])

      assert counts == expected_counts
    end
  end

  describe "errors" do
    test "wrong input rank" do
      assert_raise ArgumentError,
                   "expected tensor to have shape {num_documents, num_tokens}, got tensor with shape: {3}",
                   fn ->
                     CountVectorizer.fit_transform(Nx.tensor([1, 2, 3]))
                   end
    end
  end
end
