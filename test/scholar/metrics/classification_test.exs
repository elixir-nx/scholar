defmodule Scholar.Metrics.ClassificationTest do
  use Scholar.Case, async: true

  alias Scholar.Metrics.Classification
  doctest Classification

  test "roc_curve - y_score with repeated elements" do
    y_score = Nx.tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
    y_true = Nx.tensor([0, 0, 1, 1, 1, 1])
    distinct_value_indices = Classification.distinct_value_indices(y_score)

    {fpr, tpr, thresholds} = Classification.roc_curve(y_true, y_score, distinct_value_indices)
    assert_all_close(fpr, Nx.tensor([0.0, 0.0, 0.0, 1.0]))
    assert_all_close(tpr, Nx.tensor([0.0, 0.5, 1.0, 1.0]))
    assert_all_close(thresholds, Nx.tensor([1.3, 0.3, 0.2, 0.1]))
  end

  describe "fbeta_score" do
    test "equals recall when beta is infinity" do
      beta = Nx.tensor(:infinity)
      y_true = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :u32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], type: :u32)
      fbeta_scores = Classification.fbeta_score(y_true, y_pred, beta, num_classes: 2)

      assert_all_close(fbeta_scores, Classification.recall(y_true, y_pred, num_classes: 2))
    end

    test "equals precision when beta is 0" do
      beta = 0
      y_true = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :u32)
      y_pred = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], type: :u32)
      fbeta_scores = Classification.fbeta_score(y_true, y_pred, beta, num_classes: 2)

      assert_all_close(fbeta_scores, Classification.precision(y_true, y_pred, num_classes: 2))
    end
  end
end
