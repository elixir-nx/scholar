defmodule Scholar.Preprocessing.CountVectorizer do
  use Scholar.Case, async: true
  alias Scholar.FeatureExtraction.CountVectorizer
  doctest CountVectorizer

  describe "fit_transform" do
    test "without padding" do
      tesnsor = Nx.tensor([[2, 3, 0], [1, 4, 4]])

      counts =
        CountVectorizer.fit_transform(tesnsor,
          max_token_id: CountVectorizer.max_token_id(tesnsor)
        )

      expected_counts = Nx.tensor([[1, 0, 1, 1, 0], [0, 1, 0, 0, 2]])

      assert counts == expected_counts
    end

    test "with padding" do
      tensor = Nx.tensor([[2, 3, 0], [1, 4, -1]])

      counts =
        CountVectorizer.fit_transform(tensor, max_token_id: CountVectorizer.max_token_id(tensor))

      expected_counts = Nx.tensor([[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]])

      assert counts == expected_counts
    end
  end

  describe "max_token_id" do
    test "without padding" do
      tensor = Nx.tensor([[2, 3, 0], [1, 4, 4]])
      assert CountVectorizer.max_token_id(tensor) == 4
    end

    test "with padding" do
      tensor = Nx.tensor([[2, 3, 0], [1, 4, -1]])
      assert CountVectorizer.max_token_id(tensor) == 4
    end
  end

  describe "errors" do
    test "wrong input rank" do
      assert_raise ArgumentError,
                   "expected tensor to have shape {num_documents, num_tokens}, got tensor with shape: {3}",
                   fn ->
                     CountVectorizer.fit_transform(Nx.tensor([1, 2, 3]), max_token_id: 3)
                   end
    end
  end
end
