defmodule Scholar.FeatureExtraction.CountVectorizer do
  @moduledoc """
  A `CountVectorizer` converts already indexed collection of text documents to a matrix of token counts.
  """
  import Nx.Defn

  opts_schema = [
    max_token_id: [
      type: :pos_integer,
      required: true,
      doc: ~S"""
      Maximum token id in the input tensor.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Generates a count matrix where each row corresponds to a document in the input corpus, and each column corresponds to a unique token in the vocabulary of the corpus.

  The input must be a 2D tensor where:
  * Each row represents a document.
  * Each document has integer values representing tokens.

  The same number represents the same token in the vocabulary. Tokens should start from 0 and be consecutive. Negative values are ignored, making them suitable for padding.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Examples
      iex> t = Nx.tensor([[0, 1, 2], [1, 3, 4]])
      iex> Scholar.FeatureExtraction.CountVectorizer.fit_transform(t, max_token_id: Scholar.FeatureExtraction.CountVectorizer.max_token_id(t))
      Nx.tensor([
          [1, 1, 1, 0, 0],
          [0, 1, 0, 1, 1]
        ])

  With padding:
      iex> t = Nx.tensor([[0, 1, -1], [1, 3, 4]])
      iex> Scholar.FeatureExtraction.CountVectorizer.fit_transform(t, max_token_id: Scholar.FeatureExtraction.CountVectorizer.max_token_id(t))
      Nx.tensor([
            [1, 1, 0, 0, 0],
            [0, 1, 0, 1, 1]
        ])
  """
  deftransform fit_transform(tensor, opts \\ []) do
    fit_transform_n(tensor, NimbleOptions.validate!(opts, @opts_schema))
  end

  @doc """
  Computes the max_token_id option from given tensor.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> Scholar.FeatureExtraction.CountVectorizer.max_token_id(t)
      2
  """
  def max_token_id(tensor) do
    tensor |> Nx.reduce_max() |> Nx.to_number()
  end

  defnp fit_transform_n(tensor, opts) do
    check_for_rank(tensor)
    counts = Nx.broadcast(0, {Nx.axis_size(tensor, 0), opts[:max_token_id] + 1})

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

    counts
  end

  defnp check_for_rank(tensor) do
    if Nx.rank(tensor) != 2 do
      raise ArgumentError,
            """
            expected tensor to have shape {num_documents, num_tokens}, \
            got tensor with shape: #{inspect(Nx.shape(tensor))}\
            """
    end
  end
end
