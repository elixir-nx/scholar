defmodule Scholar.Metrics.DiscountedCumulativeGain do
  @moduledoc """
  Discounted Cumulative Gain (DCG) is a measure of ranking quality.
  It is based on the assumption that highly relevant documents appearing lower
  in a search result list should be penalized as the graded relevance value is
  reduced logarithmically proportional to the position of the result.
  """

  @doc """
  Computes the DCG based on true relevance scores (`y_true`) and their respective predicted scores (`y_score`).
  """
  def compute(y_true, y_score, k \\ nil) do
    if Nx.shape(y_true) != Nx.shape(y_score) do
      raise ArgumentError, "y_true and y_score tensors must have the same shape"
    end

    {adjusted_y_true, adjusted_y_score} = handle_ties(y_true, y_score)

    sorted_indices = Nx.argsort(adjusted_y_score, axis: 0, direction: :desc)
    sorted_y_true = Nx.take(adjusted_y_true, sorted_indices)

    truncated_y_true = truncate_at_k(sorted_y_true, k)
    dcg_value(truncated_y_true)
  end

  defp handle_ties(y_true, y_score) do
    sorted_y_true = Nx.sort(y_true, axis: 0, direction: :desc)
    sorted_y_score = Nx.sort(y_score, axis: 0, direction: :desc)

    diff = Nx.diff(sorted_y_score)
    selector = Nx.pad(diff, 1, [{1, 0, 0}])
    adjusted_y_score = Nx.select(selector, sorted_y_score, 0)

    adjusted_y_true = Nx.select(selector, sorted_y_true, 0)

    {adjusted_y_true, adjusted_y_score}
  end

  defp dcg_value(y_true) do
    float_y_true = Nx.as_type(y_true, :f32)

    log_tensor =
      y_true
      |> Nx.shape()
      |> Nx.iota()
      |> Nx.as_type(:f32)
      |> Nx.add(2.0)
      |> Nx.log2()

    if Enum.any?(Nx.to_flat_list(log_tensor), &(&1 < 0 or &1 !== &1)) do
      raise ArithmeticError, "Encountered -Inf or NaN in log_tensor during DCG computation"
    end

    div_result = Nx.divide(float_y_true, log_tensor)

    Nx.sum(div_result)
  end

  defp truncate_at_k(tensor, nil), do: tensor

  defp truncate_at_k(tensor, k) do
    shape = Nx.shape(tensor)

    if Tuple.to_list(shape) |> Enum.at(0) > k do
      {top_k, _rest} = Nx.split(tensor, k, axis: 0)
      top_k
    else
      tensor
    end
  end
end
