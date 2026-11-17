defmodule Scholar.Metrics.Ranking do
  @moduledoc """
  Provides metrics and calculations related to ranking quality.

  Ranking metrics evaluate the quality of ordered lists of items,
  often used in information retrieval and recommendation systems.

  This module currently supports the following ranking metrics:
    * Discounted Cumulative Gain (DCG)
  """

  import Nx.Defn
  import Scholar.Shared

  @ln2 :math.log(2)

  @dcg_opts [
    k: [
      type: {:or, [:pos_integer, nil]},
      doc: "Truncation parameter to consider only the top-k elements."
    ]
  ]

  @dcg_opts_schema NimbleOptions.new!(@dcg_opts)

  @doc """
  Computes the discounted cumulative gain (DCG) based on true relevance scores `y_true`
  and their respective predicted scores `y_score`.

  Tied scores get the mean gain of their group, which is the expected DCG when the ties
  are broken at random. This follows scikit-learn and McSherry and Najork, "Computing
  Information Retrieval Performance Measures Efficiently in the Presence of Tied Scores"
  (ECIR 2008).

  ## Options

  #{NimbleOptions.docs(@dcg_opts_schema)}

  ## Examples

      iex> y_true = Nx.tensor([10, 0, 0, 1, 5])
      iex> y_score = Nx.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
      iex> Scholar.Metrics.Ranking.dcg(y_true, y_score, k: 1)
      #Nx.Tensor<
        f32
        7.5
      >
  """
  deftransform dcg(y_true, y_score, opts \\ []) do
    dcg_n(y_true, y_score, opts)
  end

  @doc """
  Same as `dcg/3`.
  """
  defn dcg_n(y_true, y_score, opts) do
    opts = validate_opts(opts)

    y_true_shape = Nx.shape(y_true)
    y_score_shape = Nx.shape(y_score)

    check_shape(y_true_shape, y_score_shape)

    type = Nx.Type.merge(to_float_type(y_true), to_float_type(y_score))
    gains = tie_averaged_gains(y_true, y_score, type)
    Nx.sum(gains * discount(Nx.axis_size(gains, 0), type, opts))
  end

  deftransformp validate_opts(opts) do
    NimbleOptions.validate!(opts, @dcg_opts_schema)
  end

  defnp check_shape(y_true, y_pred) do
    assert_same_shape!(y_true, y_pred)
  end

  defnp tie_averaged_gains(y_true, y_score, type) do
    size = Nx.axis_size(y_true, 0)
    order = Nx.argsort(y_score, direction: :desc)
    gains = y_true |> Nx.take(order) |> Nx.as_type(type)
    scores = Nx.take(y_score, order)

    previous_scores = Nx.take(scores, Nx.max(Nx.iota({size}) - 1, 0))
    groups = Nx.cumulative_sum(Nx.as_type(scores != previous_scores, :s32))
    indices = Nx.new_axis(groups, 1)

    group_gains = Nx.indexed_add(Nx.broadcast(Nx.tensor(0, type: type), {size}), indices, gains)
    group_sizes = Nx.indexed_add(Nx.broadcast(0, {size}), indices, Nx.broadcast(1, {size}))

    Nx.take(group_gains / Nx.max(group_sizes, 1), groups)
  end

  defnp discount(size, type, opts) do
    positions = Nx.iota({size}, type: type)
    # Nx.log2 computes log(2) in f32 in Nx 1.0, which loses precision for f64 inputs
    discount = @ln2 / Nx.log(positions + 2)

    case opts[:k] do
      nil -> discount
      k -> Nx.select(positions < k, discount, 0)
    end
  end

  @doc """
  Computes the normalized discounted cumulative gain (NDCG) based on true relevance scores `y_true` and their respective predicted scores `y_score`.

  Tied scores are handled as in `dcg/3`.

  ## Options

  #{NimbleOptions.docs(@dcg_opts_schema)}

  ## Examples

      iex> true_relevance = Nx.tensor([10, 0, 0, 1, 5])
      iex> scores = Nx.tensor([0.1, 0.2, 0.3, 4, 70])
      iex> Scholar.Metrics.Ranking.ndcg_n(true_relevance, scores)
      #Nx.Tensor<
        f32
        0.695694
      >
      iex> scores = Nx.tensor([0.05, 1.1, 1.0, 0.5, 0.0])
      iex> Scholar.Metrics.Ranking.ndcg_n(true_relevance, scores)
      #Nx.Tensor<
        f32
        0.4936802
      >
      iex> scores = Nx.tensor([0.05, 1.1, 1.0, 0.5, 0.0])
      iex> Scholar.Metrics.Ranking.ndcg_n(true_relevance, scores, k: 4)
      #Nx.Tensor<
        f32
        0.3520241
      >
      iex> scores = Nx.tensor([1, 0, 0, 0, 1])
      iex> Scholar.Metrics.Ranking.ndcg_n(true_relevance, scores, k: 1)
      #Nx.Tensor<
        f32
        0.75
      >
      iex> Scholar.Metrics.Ranking.ndcg_n(true_relevance, true_relevance, k: 4)
      #Nx.Tensor<
        f32
        1.0
      >
  """
  defn ndcg_n(y_true, y_score, opts \\ []) do
    dcg_n(y_true, y_score, opts) / dcg_n(y_true, y_true, opts)
  end
end
