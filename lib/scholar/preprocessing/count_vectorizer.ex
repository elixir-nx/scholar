defmodule Scholar.Preprocessing.CountVectorizer do
  defstruct vocabulary: nil,
            fixed_vocabulary: false,
            ngram_range: {1, 1},
            max_features: nil,
            min_df: 1.0,
            max_df: 1.0,
            stop_words: nil,
            binary: false

  defp make_ngrams(tokens, ngram_range) do
    {min_n, max_n} = ngram_range
    n_original_tokens = length(tokens)

    ngrams =
      for n <- min_n..min(max_n, n_original_tokens) do
        for i <- 0..(n_original_tokens - n) do
          Enum.slice(tokens, i, n) |> Enum.join(" ")
        end
      end

    ngrams |> Enum.flat_map(& &1)
  end

  defp do_process(doc, opts) do
    {pre_mod, pre_func, pre_args} = opts[:preprocessor]
    {token_mod, token_func, token_args} = opts[:tokenizer]

    doc
    |> then(fn doc -> apply(pre_mod, pre_func, [doc | pre_args]) end)
    |> then(fn doc -> apply(token_mod, token_func, [doc | token_args]) end)
    |> make_ngrams(opts[:ngram_range])
    |> Enum.filter(fn token ->
      if not is_nil(opts[:stop_words]), do: token not in opts[:stop_words], else: true
    end)
  end

  def build_vocab(corpus, opts \\ []) do
    case opts[:vocabulary] do
      nil ->
        corpus
        |> Enum.reduce([], fn doc, vocab ->
          vocab ++ (doc |> do_process(opts))
        end)
        |> Enum.sort()
        |> Enum.with_index()
        |> Enum.into(%{})

      _ ->
        opts[:vocabulary]
    end
  end

  def new(corpus, opts \\ []) do
    opts = Scholar.Preprocessing.Shared.validate_shared!(opts)
    fixed_vocab = if is_nil(opts[:vocabulary]), do: false, else: true
    vocab = build_vocab(corpus, opts)
    struct(__MODULE__, vocabulary: vocab, fixed_vocabulary: fixed_vocab)
  end
end
