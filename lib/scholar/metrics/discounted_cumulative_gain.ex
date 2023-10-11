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
    # Ensure tensors are of the same shape
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
    # Zip y_true and y_score together to work with pairs and convert to lists
    zipped = y_true |> Nx.to_list() |> Enum.zip(Nx.to_list(y_score))

    # Group items by their predicted scores and adjust groups if they contain ties
    adjusted =
      zipped
      |> Enum.group_by(&elem(&1, 1))
      |> Enum.flat_map(&adjust_group/1)

    # Convert the lists back to tensors
    {
      Nx.tensor(Enum.map(adjusted, &elem(&1, 0))),
      Nx.tensor(Enum.map(adjusted, &elem(&1, 1)))
    }
  end

  # If a group has more than one element (i.e., there are ties), sort it by true_val
  # and assign all elements the average rank. Otherwise, return the group unmodified.
  defp adjust_group({_score, [single]}), do: [single]

  defp adjust_group({score, group}) when length(group) > 1 do
    group
    |> Enum.sort_by(&elem(&1, 0), &>=/2)
    |> Enum.map(&{elem(&1, 0), score})
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
