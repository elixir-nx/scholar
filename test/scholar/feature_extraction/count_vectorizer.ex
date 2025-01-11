defmodule Scholar.Preprocessing.BinarizerTest do
  use Scholar.Case, async: true
  alias Scholar.FeatureExtraction.CountVectorizer
  doctest CountVectorizer

  describe "fit_transform" do
    test "fit_transform test - default options" do
      result = CountVectorizer.fit_transform(["i love elixir", "hello world"])

      expected_counts =
        Nx.tensor([
          [1, 0, 1, 1, 0],
          [0, 1, 0, 0, 1]
        ])

      expected_vocabulary = %{
        "elixir" => Nx.tensor(0),
        "hello" => Nx.tensor(1),
        "i" => Nx.tensor(2),
        "love" => Nx.tensor(3),
        "world" => Nx.tensor(4)
      }

      assert result.counts == expected_counts
      assert result.vocabulary == expected_vocabulary
    end

    test "fit_transform test - removes interpunction" do
      result = CountVectorizer.fit_transform(["i love elixir.", "hello, world!"])

      expected_counts =
        Nx.tensor([
          [1, 0, 1, 1, 0],
          [0, 1, 0, 0, 1]
        ])

      expected_vocabulary = %{
        "elixir" => Nx.tensor(0),
        "hello" => Nx.tensor(1),
        "i" => Nx.tensor(2),
        "love" => Nx.tensor(3),
        "world" => Nx.tensor(4)
      }

      assert result.counts == expected_counts
      assert result.vocabulary == expected_vocabulary
    end

    test "fit_transform test - ignores case" do
      result = CountVectorizer.fit_transform(["i love elixir", "hello world HELLO"])

      expected_counts =
        Nx.tensor([
          [1, 0, 1, 1, 0],
          [0, 2, 0, 0, 1]
        ])

      expected_vocabulary = %{
        "elixir" => Nx.tensor(0),
        "hello" => Nx.tensor(1),
        "i" => Nx.tensor(2),
        "love" => Nx.tensor(3),
        "world" => Nx.tensor(4)
      }

      assert result.counts == expected_counts
      assert result.vocabulary == expected_vocabulary
    end

    test "fit_transform test - already indexed tensor" do
      result =
        CountVectorizer.fit_transform(
          Nx.tensor([
            [2, 3, 0],
            [1, 4, 4]
          ]),
          indexed_tensor: true
        )

      expected_counts =
        Nx.tensor([
          [1, 0, 1, 1, 0],
          [0, 1, 0, 0, 2]
        ])

      assert result.counts == expected_counts
      assert result.vocabulary == %{}
    end

    test "fit_transform test - already indexed tensor with padding" do
      result =
        CountVectorizer.fit_transform(
          Nx.tensor([
            [2, 3, 0],
            [1, 4, -1]
          ]),
          indexed_tensor: true
        )

      expected_counts =
        Nx.tensor([
          [1, 0, 1, 1, 0],
          [0, 1, 0, 0, 1]
        ])

      assert result.counts == expected_counts
      assert result.vocabulary == %{}
    end
  end
end
