defmodule Scholar.MetricsTest do
  use Scholar.Case, async: true

  doctest Scholar.Metrics

  test "roc_curve - y_score with repeated elements" do
    y_score = Nx.tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
    y_true = Nx.tensor([0, 0, 1, 1, 1, 1])
    distinct_value_indices = Scholar.Metrics.distinct_value_indices(y_score)

    {fpr, tpr, thresholds} = Scholar.Metrics.roc_curve(y_true, y_score, distinct_value_indices)
    assert_all_close(fpr, Nx.tensor([0.0, 0.0, 0.0, 1.0]))
    assert_all_close(tpr, Nx.tensor([0.0, 0.5, 1.0, 1.0]))
    assert_all_close(thresholds, Nx.tensor([1.3, 0.3, 0.2, 0.1]))
  end
end
