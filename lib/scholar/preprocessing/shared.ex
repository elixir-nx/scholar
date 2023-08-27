defmodule Scholar.Preprocessing.Shared do
  @moduledoc false
  vectorizer_schema_opts = [
    ngram_range: [
      type: {:custom, __MODULE__, :validate_ngram_range, []},
      default: {1, 1},
      doc: """
      The lower and upper boundary of the range of n-values for different n-grams to be extracted.
      All values of n such that min_n <= n <= max_n will be used.
      For example an `ngram_range` of `{1, 1}` means only unigrams, `{1, 2}` means unigrams and bigrams,
      and `{2, 2}` means only bigrams.
      Only applies if `analyzer` is not callable.
      """
    ],
    max_features: [
      type: :integer,
      default: -1,
      doc: """
      If not `nil`, build a vocabulary that only consider the top `max_features` ordered by term frequency across the corpus.
      This parameter is ignored if `vocabulary` is not `nil`.
      """
    ],
    min_df: [
      type: {:custom, __MODULE__, :in_range, [:closed, 0, 1, :closed]},
      default: 1.0,
      doc: """
      When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
      This value is also called cut-off in the literature.
      If float, the parameter represents a proportion of documents, integer absolute counts.
      This parameter is ignored if `vocabulary` is not `nil`.
      """
    ],
    max_df: [
      type: {:custom, __MODULE__, :in_range, [:closed, 0, 1, :closed]},
      default: 1.0,
      doc: """
      When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold.
      This value is also called cut-off in the literature.
      If float, the parameter represents a proportion of documents, integer absolute counts.
      This parameter is ignored if `vocabulary` is not `nil`.
      """
    ],
    stop_words: [
      type: {:list, :string},
      default: [],
      doc: """
      If `stop_words` is `nil`, no stop words will be used.
      If `stop_words` is `:english`, a built-in stop word list for English is used.
      If `stop_words` is a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
      Only applies if `analyzer` is not callable.
      """
    ],
    binary: [
      type: :boolean,
      default: false,
      doc: """
      If `true`, all non-zero counts are set to 1.
      This is useful for discrete probabilistic models that model binary events rather than integer counts.
      Only applies if `analyzer` is not callable.
      """
    ],
    dtype: [
      type:
        {:in,
         [:u8, :u16, :u32, :u64, :s8, :s16, :s32, :s64, :f16, :f32, :f64, :bf16, :c64, :c128]},
      default: :f64,
      doc: """
      Type of the matrix returned by `fit_transform` or `transform`.
      Only applies if `analyzer` is not callable.
      """
    ],
    tokenizer: [
      type: :mfa,
      default: {String, :split, []},
      doc: """
      Provide the tokenization function to use on the corpus.
      The n-grams generated will use the tokens produced by the tokenizer.
      Must be in MFA format (e.g. `{Module, :function, arity}`).
      If `tokenizer` is `nil`, `String.split/2` is used.
      """
    ],
    preprocessor: [
      type: :mfa,
      default: {__MODULE__, :default_preprocessor, []},
      doc: """
      Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.
      Must be in MFA format (e.g. `{Module, :function, arity}`).
      Default performs `String.downcase/1` |> `String.normalize(:nfkd)`.
      """
    ],
    norm: [
      type: {:in, [:l1, :l2, nil]},
      default: :l2,
      doc: """
      Norm used to normalize term vectors.
      """
    ],
    vocabulary: [
      type: {:custom, __MODULE__, :validate_vocabulary, []},
      default: nil,
      doc: """
      Either a map where keys are terms and values are indices in the feature matrix, or a list of terms.
      If `vocabulary` is `nil`, a vocabulary is determined from the input documents.
      Indices in the vocabulary are expected to be unique.
      """
    ]
  ]

  @vectorizer_schema NimbleOptions.new!(vectorizer_schema_opts)

  def validate_vocabulary(vocabulary) do
    case vocabulary do
      nil ->
        {:ok, nil}

      %MapSet{} ->
        {:ok, vocabulary |> Enum.sort() |> Enum.with_index() |> Enum.into(%{})}

      _ when is_list(vocabulary) ->
        {:ok, vocabulary |> Enum.sort() |> Enum.with_index() |> Enum.into(%{})}

      _ when is_map(vocabulary) ->
        indices = vocabulary |> Map.values() |> MapSet.new()

        unless Enum.count(indices) == Enum.count(vocabulary),
          do: {:error, "vocabulary indices must be unique"}

        for i <- 0..(Enum.count(vocabulary) - 1) do
          unless Map.has_key?(vocabulary, i),
            do: {:error, "Vocabulary of size #{Enum.count(vocabulary)} missing index #{i}"}
        end

        {:ok, vocabulary}

      _ ->
        {:error, "vocabulary must be of type Map, MapSet, or List"}
    end
  end

  def default_preprocessor(text) do
    text
    |> String.downcase()
    |> String.normalize(:nfkd)
  end

  def validate_shared!(opts) do
    NimbleOptions.validate!(opts, @vectorizer_schema)
  end

  def validate_ngram_range(value = {min, max}) do
    if min <= max, do: {:ok, value}, else: {:error, "min must be less than or equal to max"}
  end

  def validate_ngram_range(value) do
    unless is_tuple(value) and tuple_size(value) == 2,
      do: {:error, "ngram_range must be a tuple of length 2"}
  end

  def in_range(value, left_bracket, min, max, right_bracket) do
    in_range? =
      case {left_bracket, min, max, right_bracket} do
        {_, nil, nil, _} ->
          true

        {_, nil, max, :closed} ->
          value <= max

        {_, nil, max, :open} ->
          value < max

        {:closed, min, nil, _} ->
          value >= min

        {:open, min, nil, _} ->
          value > min

        {:closed, min, max, :closed} ->
          value >= min and value <= max

        {:open, min, max, :closed} ->
          value > min and value <= max

        {:closed, min, max, :open} ->
          value >= min and value < max

        {:open, min, max, :open} ->
          value > min and value < max
      end

    if in_range?, do: {:ok, value}, else: {:error, "Value #{value} is not in range"}
  end
end
