defmodule Scholar.Metrics.ClassificationTest do
  use Scholar.Case, async: true

  alias Scholar.Metrics.Classification
  doctest Classification

  test "confusion_matrix - sample weights as a tensor match the same weights as a list" do
    y_true = Nx.tensor([0, 0, 1, 1], type: :u32)
    y_pred = Nx.tensor([0, 1, 1, 1], type: :u32)

    from_list =
      Classification.confusion_matrix(y_true, y_pred,
        num_classes: 2,
        sample_weights: [1, 2, 1, 1]
      )

    from_tensor =
      Classification.confusion_matrix(y_true, y_pred,
        num_classes: 2,
        sample_weights: Nx.tensor([1, 2, 1, 1])
      )

    assert from_tensor == from_list
  end

  test "confusion_matrix - a scalar sample weight applies to every sample" do
    y_true = Nx.tensor([0, 0, 1, 1], type: :u32)
    y_pred = Nx.tensor([0, 1, 1, 1], type: :u32)

    weighted =
      Classification.confusion_matrix(y_true, y_pred,
        num_classes: 2,
        sample_weights: Nx.tensor(2)
      )

    unweighted = Classification.confusion_matrix(y_true, y_pred, num_classes: 2)

    assert Nx.to_flat_list(weighted) == Nx.to_flat_list(Nx.multiply(unweighted, 2))
  end

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

  describe "mcc/2" do
    test "returns 1 for perfect predictions" do
      y_true = Nx.tensor([1, 0, 1, 0, 1])
      y_pred = Nx.tensor([1, 0, 1, 0, 1])
      assert Classification.mcc(y_true, y_pred) == Nx.tensor([1.0], type: :f32)
    end

    test "returns -1 for completely wrong predictions" do
      y_true = Nx.tensor([1, 0, 1, 0, 1])
      y_pred = Nx.tensor([0, 1, 0, 1, 0])
      assert Classification.mcc(y_true, y_pred) == Nx.tensor([-1.0], type: :f32)
    end

    test "returns 0 when all predictions are positive" do
      y_true = Nx.tensor([1, 0, 1, 0, 1])
      y_pred = Nx.tensor([1, 1, 1, 1, 1])
      assert Classification.mcc(y_true, y_pred) == Nx.tensor([0.0], type: :f32)
    end

    test "returns 0 when all predictions are negative" do
      y_true = Nx.tensor([1, 0, 1, 0, 1])
      y_pred = Nx.tensor([0, 0, 0, 0, 0])
      assert Classification.mcc(y_true, y_pred) == Nx.tensor([0.0], type: :f32)
    end

    test "computes MCC for generic case" do
      y_true = Nx.tensor([1, 0, 1, 0, 1])
      y_pred = Nx.tensor([1, 0, 1, 1, 1])
      assert Classification.mcc(y_true, y_pred) == Nx.tensor([0.6123723983764648], type: :f32)
    end

    test "returns 0 when TP, TN, FP, and FN are all 0" do
      y_true = Nx.tensor([0, 0, 0, 0, 0])
      y_pred = Nx.tensor([0, 0, 0, 0, 0])
      assert Classification.mcc(y_true, y_pred) == Nx.tensor([0.0], type: :f32)
    end
  end
end
