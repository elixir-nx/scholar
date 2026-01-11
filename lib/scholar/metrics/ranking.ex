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

  @dcg_opts [
    k: [
      type: {:custom, Scholar.Options, :positive_number, []},
      doc: "Truncation parameter to consider only the top-k elements."
    ]
  ]

  @dcg_opts_schema NimbleOptions.new!(@dcg_opts)

  deftransform dcg(y_true, y_score, opts \\ []) do
    dcg_n(y_true, y_score, NimbleOptions.validate!(opts, @dcg_opts_schema))
  end

  @doc """
  ## Options
  #{NimbleOptions.docs(@dcg_opts_schema)}

  Computes the DCG based on true relevance scores (`y_true`) and their respective predicted scores (`y_score`).
  """
  defn dcg_n(y_true, y_score, opts) do
    y_true_shape = Nx.shape(y_true)
    y_score_shape = Nx.shape(y_score)

    check_shape(y_true_shape, y_score_shape)

    {adjusted_y_true, adjusted_y_score} = handle_ties(y_true, y_score)

    sorted_indices = Nx.argsort(adjusted_y_score, axis: 0, direction: :desc)
    sorted_y_true = Nx.take(adjusted_y_true, sorted_indices)

    truncated_y_true = truncate_at_k(sorted_y_true, opts)
    dcg_value(truncated_y_true)
  end

  defnp check_shape(y_true, y_pred) do
    assert_same_shape!(y_true, y_pred)
  end

  defnp handle_ties(y_true, y_score) do
    sorted_indices = Nx.argsort(y_score, axis: 0, direction: :desc)

    sorted_y_true = Nx.take(y_true, sorted_indices)
    sorted_y_score = Nx.take(y_score, sorted_indices)

    tie_sorted_indices = Nx.argsort(sorted_y_true, axis: 0, direction: :desc)
    adjusted_y_true = Nx.take(sorted_y_true, tie_sorted_indices)
    adjusted_y_score = Nx.take(sorted_y_score, tie_sorted_indices)

    {adjusted_y_true, adjusted_y_score}
  end

  defnp dcg_value(y_true) do
    float_y_true = Nx.as_type(y_true, :f32)

    log_tensor =
      y_true
      |> Nx.shape()
      |> Nx.iota()
      |> Nx.as_type(:f32)
      |> Nx.add(2.0)
      |> Nx.log2()

    div_result = Nx.divide(float_y_true, log_tensor)

    Nx.sum(div_result)
  end

  defnp truncate_at_k(tensor, opts) do
    case opts[:k] do
      nil ->
        tensor

      _ ->
        if opts[:k] > Nx.axis_size(tensor, 0) do
          tensor
        else
          {top_k, _rest} = Nx.split(tensor, opts[:k], axis: 0)
          top_k
        end
    end
  end

  @doc """
  Computes the normalized discounted cumulative gain (NDCG) based on true relevance scores `y_true` and their respective predicted scores `y_score`.

  ## Options

  #{NimbleOptions.docs(@dcg_opts_schema)}

  ## Examples

      iex> true_relevance = Nx.tensor([10, 0, 0, 1, 5])
      iex> scores = Nx.tensor([0.1, 0.2, 0.3, 4, 70])
      iex> Scholar.Metrics.Ranking.ndcg_n(true_relevance, scores)
      #Nx.Tensor<
        f32
        0.6956940293312073
      >
      iex> scores = Nx.tensor([0.05, 1.1, 1.0, 0.5, 0.0])
      iex> Scholar.Metrics.Ranking.ndcg_n(true_relevance, scores)
      #Nx.Tensor<
        f32
        0.4936802089214325
      >
      iex> scores = Nx.tensor([0.05, 1.1, 1.0, 0.5, 0.0])
      iex> Scholar.Metrics.Ranking.ndcg_n(true_relevance, scores, k: 4)
      #Nx.Tensor<
        f32
        0.352024108171463
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
