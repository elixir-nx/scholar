defmodule Scholar.Metrics.MCC do
  @moduledoc """
  Matthews Correlation Coefficient (MCC) provides a measure of the quality of binary classifications.
  It returns a value between -1 and 1 where 1 represents a perfect prediction, 0 represents no better
  than random prediction, and -1 indicates total disagreement between prediction and observation.
  """

  import Nx.Defn

  @doc """
  Computes the Matthews Correlation Coefficient (MCC) for binary classification.
  Assumes `y_true` and `y_pred` are binary tensors (0 or 1).
  """
  defn compute(y_true, y_pred) do
    true_positives = calculate_true_positives(y_true, y_pred)
    true_negatives = calculate_true_negatives(y_true, y_pred)
    false_positives = calculate_false_positives(y_true, y_pred)
    false_negatives = calculate_false_negatives(y_true, y_pred)

    mcc_numerator = true_positives * true_negatives - false_positives * false_negatives

    mcc_denominator =
      Nx.sqrt(
        (true_positives + false_positives) *
          (true_positives + false_negatives) *
          (true_negatives + false_positives) *
          (true_negatives + false_negatives)
      )

    zero_tensor = Nx.tensor([0.0], type: :f32)

    if Nx.all(
           true_positives == zero_tensor and
           true_negatives ==  zero_tensor
       ) do
      Nx.tensor([-1.0], type: :f32)
    else
      Nx.select(
        mcc_denominator == zero_tensor,
        zero_tensor,
        mcc_numerator / mcc_denominator
      )
    end
  end

  defnp calculate_true_positives(y_true, y_pred) do
    Nx.sum(Nx.equal(y_true, 1) * Nx.equal(y_pred, 1))
  end

  defnp calculate_true_negatives(y_true, y_pred) do
    Nx.sum(Nx.equal(y_true, 0) * Nx.equal(y_pred, 0))
  end

  defnp calculate_false_positives(y_true, y_pred) do
    Nx.sum(Nx.equal(y_true, 0) * Nx.equal(y_pred, 1))
  end

  defnp calculate_false_negatives(y_true, y_pred) do
    Nx.sum(Nx.equal(y_true, 1) * Nx.equal(y_pred, 0))
  end
end
