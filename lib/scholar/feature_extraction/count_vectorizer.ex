defmodule Scholar.FeatureExtraction.CountVectorizer do
  @moduledoc """
  A `CountVectorizer` converts a collection of text documents to a matrix of token counts.

  Each row of the matrix corresponds to a document in the input corpus, and each column corresponds to a unique token from the vocabulary of the corpus.

  Supports also already indexed tensors.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:counts, :vocabulary]}
  defstruct [:counts, :vocabulary]

  binarize_schema = [
    indexed_tensor: [
      type: :boolean,
      default: false,
      doc: ~S"""
      If set to true, it assumes the input corpus is already an indexed tensor
      instead of raw text strings. This skips preprocessing and vocabulary creation.
      Tensors needs to be homogeneous, so you can pad them with -1, and they will be ignored.
      """
    ]
  ]

  @binarize_schema NimbleOptions.new!(binarize_schema)

  @doc """
  Processes the input corpus and generates a count matrix and vocabulary.

  This function performs:

  * tokenization of input text (splitting by whitespace and removing punctuation)
  * vocabulary construction
  * creation of a count tensor

  ## Options

  #{NimbleOptions.docs(@binarize_schema)}

  ## Examples

      iex> corpus = ["Elixir is amazing!", "Elixir provides great tools."]
      iex> Scholar.FeatureExtraction.CountVectorizer.fit_transform(corpus)
      %Scholar.FeatureExtraction.CountVectorizer{
        counts: Nx.tensor(
          [
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 1]
          ]
        ),
        vocabulary: %{
          "amazing" => Nx.tensor(0),
          "elixir" => Nx.tensor(1),
          "great" => Nx.tensor(2),
          "is" => Nx.tensor(3),
          "provides" => Nx.tensor(4),
          "tools" => Nx.tensor(5)
        }
      }

  Input can optionally be an indexed tensor to skip preprocessing and vocabulary creation:

      iex> t = Nx.tensor([[0, 1, 2], [1, 3, 4]])
      iex> Scholar.FeatureExtraction.CountVectorizer.fit_transform(t, indexed_tensor: true)
      %Scholar.FeatureExtraction.CountVectorizer{
        counts: Nx.tensor([
          [1, 1, 1, 0, 0],
          [0, 1, 0, 1, 1]
        ]),
        vocabulary: %{}
      }
  """
  deftransform fit_transform(corpus, opts \\ []) do
    {tensor, vocabulary} =
      if opts[:indexed_tensor] do
        {corpus, %{}}
      else
        preprocessed_corpus = preprocess(corpus)
        vocabulary = create_vocabulary(preprocessed_corpus)
        tensor = create_tensor(preprocessed_corpus, vocabulary)
        {tensor, vocabulary}
      end

    max_index = tensor |> Nx.reduce_max() |> Nx.add(1) |> Nx.to_number()

    opts =
      NimbleOptions.validate!(opts, @binarize_schema) ++
        [max_index: max_index, vocabulary: vocabulary]

    fit_transform_n(tensor, opts)
  end

  defnp fit_transform_n(tensor, opts) do
    counts = Nx.broadcast(0, {Nx.axis_size(tensor, 0), opts[:max_index]})

    {_, counts} =
      while {{i = 0, tensor}, counts}, Nx.less(i, Nx.axis_size(tensor, 0)) do
        {_, counts} =
          while {{j = 0, i, tensor}, counts}, Nx.less(j, Nx.axis_size(tensor, 1)) do
            index = tensor[i][j]

            counts =
              if Nx.any(Nx.less(index, 0)),
                do: counts,
                else: Nx.indexed_add(counts, Nx.stack([i, index]), 1)

            {{j + 1, i, tensor}, counts}
          end

        {{i + 1, tensor}, counts}
      end

    %__MODULE__{
      counts: counts,
      vocabulary: opts[:vocabulary]
    }
  end

  deftransformp preprocess(corpus) do
    corpus
    |> Enum.map(&String.downcase/1)
    |> Enum.map(&String.split(&1, ~r/\W+/, trim: true))
  end

  deftransformp create_vocabulary(preprocessed_corpus) do
    preprocessed_corpus
    |> List.flatten()
    |> Enum.uniq()
    |> Enum.sort()
    |> Enum.with_index()
    |> Enum.into(%{})
  end

  deftransformp create_tensor(preprocessed_corpus, vocabulary) do
    indexed_sublist =
      preprocessed_corpus
      |> Enum.map(fn words ->
        words
        |> Enum.map(&Map.get(vocabulary, &1, :nan))
      end)

    max_length = indexed_sublist |> Enum.map(&length/1) |> Enum.max()

    indexed_sublist
    |> Enum.map(&Enum.concat(&1, List.duplicate(-1, max_length - length(&1))))
    |> Nx.tensor()
    |> Nx.as_type({:s, 64})
  end
end
