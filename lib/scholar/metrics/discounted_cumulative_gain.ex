defmodule Scholar.Metrics.DiscountedCumulativeGain do
  @moduledoc """
  Discounted Cumulative Gain (DCG) is a measure of ranking quality.
  It is based on the assumption that highly relevant documents appearing lower
  in a search result list should be penalized as the graded relevance value is
  reduced logarithmically proportional to the position of the result.
  """

  import Nx.Defn
  import Scholar.Shared
  require Nx

  opts = [
    k: [
      default: nil,
      type: {:custom, Scholar.Options, :positive_number_or_nil, []},
      doc: "Truncation parameter to consider only the top-k elements."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  deftransform compute(y_true, y_score, opts \\ []) do
    compute_n(y_true, y_score, NimbleOptions.validate!(opts, @opts_schema))
  end

  @doc """
  ## Options
  #{NimbleOptions.docs(@opts_schema)}

  Computes the DCG based on true relevance scores (`y_true`) and their respective predicted scores (`y_score`).
  """
  defn compute_n(y_true, y_score, opts) do
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
    sorted_y_true = Nx.sort(y_true, axis: 0, direction: :desc)
    sorted_y_score = Nx.sort(y_score, axis: 0, direction: :desc)

    diff = Nx.diff(sorted_y_score)
    selector = Nx.pad(diff, 1, [{1, 0, 0}])
    adjusted_y_score = Nx.select(selector, sorted_y_score, 0)

    adjusted_y_true = Nx.select(selector, sorted_y_true, 0)

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
end
