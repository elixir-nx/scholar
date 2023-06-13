defmodule Scholar.Metrics do
  @moduledoc """
  Metric functions.

  Metrics are used to measure the performance and compare
  the performance of any kind of classifier in
  easy-to-understand terms.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  import Nx.Defn, except: [assert_shape: 2, assert_shape_pattern: 2]
  import Scholar.Shared

  general_schema = [
    num_classes: [
      required: true,
      type: :pos_integer,
      doc: "Number of classes contained in the input tensors"
    ]
  ]

  f1_score_schema =
    general_schema ++
      [
        average: [
          type: {:in, [:micro, :macro, :weighted, :none]},
          default: :none,
          doc: """
          This determines the type of averaging performed on the data.

          * `:macro` - Calculate metrics for each label, and find their unweighted mean.
          This does not take label imbalance into account.

          * `:weighted` - Calculate metrics for each label, and find their average weighted by
          support (the number of true instances for each label).

          * `:micro` - Calculate metrics globally by counting the total true positives,
          false negatives and false positives.

          * `:none` - The f1 scores for each class are returned.
          """
        ]
      ]

  @general_schema NimbleOptions.new!(general_schema)
  @f1_score_schema NimbleOptions.new!(f1_score_schema)

  # Standard Metrics

  @doc ~S"""
  Computes the accuracy of the given predictions
  for binary and multi-class classification problems.

  ## Examples

      iex> Scholar.Metrics.accuracy(Nx.tensor([1, 0, 0]), Nx.tensor([1, 0, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >
      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.accuracy(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.6000000238418579
      >
  """
  defn accuracy(y_true, y_pred) do
    check_shape(y_true, y_pred)
    Nx.mean(y_pred == y_true)
  end

  @doc ~S"""
  Computes the precision of the given predictions with respect to
  the given targets for binary classification problems.

  If the sum of true positives and false positives is 0, then the
  result is 0 to avoid zero division.

  ## Examples

      iex> Scholar.Metrics.binary_precision(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >
  """
  defn binary_precision(y_true, y_pred) do
    check_shape(y_true, y_pred)

    true_positives = binary_true_positives(y_true, y_pred)
    false_positives = binary_false_positives(y_true, y_pred)

    safe_division(true_positives, true_positives + false_positives)
  end

  @doc """
  Computes the precision of the given predictions with respect to
  the given targets for multi-class classification problems.

  If the sum of true positives and false positives is 0, then the
  result is 0 to avoid zero division.

  ## Options

  #{NimbleOptions.docs(@general_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.precision(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 1.0, 0.25]
      >
  """
  deftransform precision(y_true, y_pred, opts \\ []) do
    precision_n(y_true, y_pred, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp precision_n(y_true, y_pred, opts) do
    check_shape(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positives = Nx.take_diagonal(cm)
    false_positives = Nx.sum(cm, axes: [0]) - true_positives

    safe_division(true_positives, true_positives + false_positives)
  end

  @doc """
  Computes the recall of the given predictions with respect to
  the given targets for binary classification problems.

  If the sum of true positives and false negatives is 0, then the
  result is 0 to avoid zero division.

  ## Examples

      iex> Scholar.Metrics.binary_recall(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >
  """
  defn binary_recall(y_true, y_pred) do
    check_shape(y_true, y_pred)

    true_positives = binary_true_positives(y_true, y_pred)
    false_negatives = binary_false_negatives(y_true, y_pred)

    safe_division(true_positives, false_negatives + true_positives)
  end

  @doc """
  Computes the recall of the given predictions with respect to
  the given targets for multi-class classification problems.

  If the sum of true positives and false negatives is 0, then the
  result is 0 to avoid zero division.

  ## Options

  #{NimbleOptions.docs(@general_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.recall(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 0.5, 1.0]
      >
  """
  deftransform recall(y_true, y_pred, opts \\ []) do
    recall_n(y_true, y_pred, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp recall_n(y_true, y_pred, opts) do
    check_shape(y_pred, y_true)

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positive = Nx.take_diagonal(cm)
    false_negative = Nx.sum(cm, axes: [1]) - true_positive

    safe_division(true_positive, true_positive + false_negative)
  end

  defnp binary_true_positives(y_true, y_pred) do
    check_shape(y_true, y_pred)
    Nx.sum(y_pred == y_true and y_pred == 1)
  end

  defnp binary_false_negatives(y_true, y_pred) do
    check_shape(y_true, y_pred)
    Nx.sum(y_pred != y_true and y_pred == 0)
  end

  defnp binary_true_negatives(y_true, y_pred) do
    check_shape(y_true, y_pred)
    Nx.sum(y_pred == y_true and y_pred == 0)
  end

  defnp binary_false_positives(y_true, y_pred) do
    check_shape(y_true, y_pred)
    Nx.sum(y_pred != y_true and y_pred == 1)
  end

  @doc """
  Computes the sensitivity of the given predictions with respect
  to the given targets for binary classification problems.

  ## Examples

      iex> Scholar.Metrics.binary_sensitivity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >
  """
  defn binary_sensitivity(y_true, y_pred) do
    check_shape(y_true, y_pred)
    binary_recall(y_true, y_pred)
  end

  @doc """
  Computes the sensitivity of the given predictions with respect
  to the given targets for multi-class classification problems.

  ## Options

  #{NimbleOptions.docs(@general_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.sensitivity(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 0.5, 1.0]
      >
  """
  deftransform sensitivity(y_true, y_pred, opts \\ []) do
    sensitivity_n(y_true, y_pred, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp sensitivity_n(y_true, y_pred, opts) do
    check_shape(y_pred, y_true)
    recall(y_true, y_pred, opts)
  end

  @doc """
  Computes the specificity of the given predictions with respect
  to the given targets for binary classification problems.

  If the sum of true negatives and false positives is 0, then the
  result is 0 to avoid zero division.

  ## Examples

      iex> Scholar.Metrics.binary_specificity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.0
      >
  """
  defn binary_specificity(y_true, y_pred) do
    check_shape(y_true, y_pred)

    true_negatives = binary_true_negatives(y_true, y_pred)
    false_positives = binary_false_positives(y_true, y_pred)

    safe_division(true_negatives, false_positives + true_negatives)
  end

  @doc """
  Computes the specificity of the given predictions with respect
  to the given targets for multi-class classification problems.

  If the sum of true negatives and false positives is 0, then the
  result is 0 to avoid zero division.

  ## Options

  #{NimbleOptions.docs(@general_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.specificity(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.8571428656578064, 1.0, 0.6666666865348816]
      >
  """
  deftransform specificity(y_true, y_pred, opts \\ []) do
    specificity_n(y_true, y_pred, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp specificity_n(y_true, y_pred, opts) do
    check_shape(y_pred, y_true)

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positive = Nx.take_diagonal(cm)
    false_positive = Nx.sum(cm, axes: [0]) - true_positive
    false_negative = Nx.sum(cm, axes: [1]) - true_positive
    true_negative = Nx.sum(cm) - (false_negative + false_positive + true_positive)

    safe_division(true_negative, false_positive + true_negative)
  end

  @doc """
  Calculates the confusion matrix given rank-1 tensors which represent
  the expected (`y_true`) and predicted (`y_pred`) classes.

  ## Options

  #{NimbleOptions.docs(@general_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1, 2, 2], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 1, 0, 2, 2, 2], type: {:u, 32})
      iex> Scholar.Metrics.confusion_matrix(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        u64[3][3]
        [
          [1, 1, 0],
          [1, 0, 1],
          [0, 0, 2]
        ]
      >
  """
  deftransform confusion_matrix(y_true, y_pred, opts \\ []) do
    confusion_matrix_n(y_true, y_pred, NimbleOptions.validate!(opts, @general_schema))
  end

  defnp confusion_matrix_n(y_true, y_pred, opts) do
    check_shape(y_pred, y_true)

    num_classes = check_num_classes(opts[:num_classes])

    zeros = Nx.broadcast(Nx.tensor(0, type: {:u, 64}), {num_classes, num_classes})
    indices = Nx.stack([y_true, y_pred], axis: 1)
    updates = Nx.broadcast(Nx.tensor(1, type: {:u, 64}), y_true)

    Nx.indexed_add(zeros, indices, updates)
  end

  @doc """
  Calculates F1 score given rank-1 tensors which represent
  the expected (`y_true`) and predicted (`y_pred`) classes.

  If all examples are true negatives, then the result is 0 to
  avoid zero division.

  ## Options

  #{NimbleOptions.docs(@f1_score_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.f1_score(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 0.6666666865348816, 0.4000000059604645]
      >
      iex> Scholar.Metrics.f1_score(y_true, y_pred, num_classes: 3, average: :macro)
      #Nx.Tensor<
        f32
        0.5777778029441833
      >
      iex> Scholar.Metrics.f1_score(y_true, y_pred, num_classes: 3, average: :weighted)
      #Nx.Tensor<
        f32
        0.6399999856948853
      >
      iex> Scholar.Metrics.f1_score(y_true, y_pred, num_classes: 3, average: :micro)
      #Nx.Tensor<
        f32
        0.6000000238418579
      >
      iex> Scholar.Metrics.f1_score(Nx.tensor([1,0,1,0]), Nx.tensor([0, 1, 0, 1]), num_classes: 2, average: :none)
      #Nx.Tensor<
        f32[2]
        [0.0, 0.0]
      >
  """
  deftransform f1_score(y_true, y_pred, opts \\ []) do
    f1_score_n(y_true, y_pred, NimbleOptions.validate!(opts, @f1_score_schema))
  end

  defnp f1_score_n(y_true, y_pred, opts) do
    check_shape(y_pred, y_true)
    num_classes = check_num_classes(opts[:num_classes])

    case opts[:average] do
      :micro ->
        accuracy(y_true, y_pred)

      _ ->
        cm = confusion_matrix(y_true, y_pred, num_classes: num_classes)
        true_positive = Nx.take_diagonal(cm)
        false_positive = Nx.sum(cm, axes: [0]) - true_positive
        false_negative = Nx.sum(cm, axes: [1]) - true_positive

        precision = safe_division(true_positive, true_positive + false_positive)

        recall = safe_division(true_positive, true_positive + false_negative)

        per_class_f1 = safe_division(2 * precision * recall, precision + recall)

        case opts[:average] do
          :none ->
            per_class_f1

          :macro ->
            Nx.mean(per_class_f1)

          :weighted ->
            support = (y_true == Nx.iota({num_classes, 1})) |> Nx.sum(axes: [1])

            safe_division(per_class_f1 * support, Nx.sum(support))
            |> Nx.sum()
        end
    end
  end

  @doc ~S"""
  Calculates the mean absolute error of predictions
  with respect to targets.

  $$MAE = \frac{\sum_{i=1}^{n} |\hat{y_i} - y_i|}{n}$$

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Scholar.Metrics.mean_absolute_error(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.5
      >
  """
  defn mean_absolute_error(y_true, y_pred) do
    assert_same_shape!(y_true, y_pred)

    (y_true - y_pred)
    |> Nx.abs()
    |> Nx.mean()
  end

  @doc ~S"""
  Calculates the mean square error of predictions
  with respect to targets.

  $$MSE = \frac{\sum_{i=1}^{n} (\hat{y_i} - y_i)^2}{n}$$

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 2.0], [0.5, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Scholar.Metrics.mean_square_error(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.5625
      >
  """
  defn mean_square_error(y_true, y_pred) do
    diff = y_true - y_pred
    (diff * diff) |> Nx.mean()
  end

  @doc ~S"""
  Computes area under the curve (AUC) using the trapezoidal rule.

  This is a general function, given points on a curve.

  ## Examples

      iex> y = Nx.tensor([0, 0, 1, 1])
      iex> pred = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> distinct_value_indices = Scholar.Metrics.calculate_distinct_value_indices(pred)
      iex> {fpr, tpr, _thresholds} = Scholar.Metrics.roc_curve(y, pred, distinct_value_indices)
      iex> Scholar.Metrics.auc(fpr, tpr)
      #Nx.Tensor<
        f32
        0.75
      >
  """
  defn auc(x, y) do
    check_shape(x, y)
    dx = x[[1..-1//1]] - x[[0..-2//1]]
    # direction = compute_direction(dx)

    # 0 means x is neither increasing nor decreasing -> error
    direction =
      cond do
        Nx.all(dx <= 0) -> -1
        Nx.all(dx >= 0) -> 1
        true -> 0
      end

    direction * trapz1d(y, x, use_x?: true)
  end

  # TODO add support for multi-dimensional x and y and move to Nx
  defnp trapz1d(y, x, opts \\ []) do
    opts = keyword!(opts, use_x?: false, dx: 1.0, axis: -1)

    d =
      cond do
        opts[:use_x?] == false -> opts[:dx]
        opts[:use_x?] == true -> x[[1..-1//1]] - x[[0..-2//1]]
      end

    Nx.sum(d * (y[[0..-2//1]] + y[[1..-1//1]]) / 2.0)
  end

  @doc ~S"""
  It's a helper function for `Scholar.Metrics.roc_curve` and `Scholar.Metrics.roc_auc_score` functions.
  You should call it and use as follows:

    `iex> distinct_value_indices = Scholar.Metrics.calculate_distinct_value_indices(scores)`
    `iex> {fpr, tpr, thresholds} = Scholar.Metrics.roc_curve(y_true, scores, distinct_value_indices, weights)`
  """
  def calculate_distinct_value_indices(y_score) do
    desc_score_indices = Nx.argsort(y_score, direction: :desc)
    y_score = Nx.take_along_axis(y_score, desc_score_indices)

    distinct_value_indices_mask =
      Nx.not_equal(y_score[[1..-1//1]], y_score[[0..-2//1]])

    Nx.iota({Nx.size(y_score) - 1})
    |> Nx.add(1)
    |> Nx.multiply(distinct_value_indices_mask)
    |> Nx.to_flat_list()
    |> Enum.filter(fn x -> x != 0 end)
    |> Nx.tensor()
    |> Nx.subtract(1)
  end

  defnp binary_clf_curve(y_true, y_score, distinct_value_indices, sample_weights) do
    check_shape(y_true, y_score)

    desc_score_indices = Nx.argsort(y_score, direction: :desc)
    y_score = Nx.take_along_axis(y_score, desc_score_indices)
    y_true = Nx.take_along_axis(y_true, desc_score_indices)

    weight = Nx.take_along_axis(sample_weights, desc_score_indices)

    threshold_idxs =
      Nx.concatenate([distinct_value_indices, Nx.new_axis(Nx.size(y_true) - 1, -1)], axis: 0)

    tps = Nx.take(Nx.cumulative_sum(y_true * weight, axis: 0), threshold_idxs)

    fps = Nx.take(Nx.cumulative_sum((1 - y_true) * weight, axis: 0), threshold_idxs)

    {fps, tps, Nx.take(y_score, threshold_idxs)}
  end

  # TODO implement :drop_intermediate option when function Nx.diff is implemented
  @doc ~S"""
  Compute Receiver operating characteristic (ROC).

  Note: this implementation is restricted to the binary classification task.

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> distinct_value_indices = Scholar.Metrics.calculate_distinct_value_indices(scores)
      iex> weights = Nx.tensor([1, 1, 2, 2])
      iex> {fpr, tpr, thresholds} = Scholar.Metrics.roc_curve(y_true, scores, distinct_value_indices, weights)
      iex> fpr
      #Nx.Tensor<
        f32[5]
        [0.0, 0.0, 0.5, 0.5, 1.0]
      >
      iex> tpr
      #Nx.Tensor<
        f32[5]
        [0.0, 0.5, 0.5, 1.0, 1.0]
      >
      iex> thresholds
      #Nx.Tensor<
        f32[5]
        [1.7999999523162842, 0.800000011920929, 0.4000000059604645, 0.3499999940395355, 0.10000000149011612]
      >
  """
  defn roc_curve(y_true, y_score, distinct_value_indices, weights) do
    num_samples = Nx.axis_size(y_true, 0)
    weights = validate_weights(weights, num_samples, type: to_float_type(y_true))
    roc_curve_n(y_true, y_score, distinct_value_indices, weights)
  end

  @doc ~S"""
  Compute Receiver operating characteristic (ROC).

  This is equivalent to calling `Nx.roc_curve/4` with weights set to ones.
  """
  defn roc_curve(y_true, y_score, distinct_value_indices) do
    weights = Nx.broadcast(Nx.tensor(1, type: to_float_type(y_true)), y_true)
    roc_curve_n(y_true, y_score, distinct_value_indices, weights)
  end

  defnp roc_curve_n(y_true, y_score, distinct_value_indices, weights) do
    check_shape(y_true, y_score)

    {fps, tps, thresholds_unpadded} =
      binary_clf_curve(y_true, y_score, distinct_value_indices, weights)

    tpr = Nx.broadcast(Nx.tensor(0, type: Nx.type(tps)), {Nx.size(tps) + 1})
    fpr = Nx.broadcast(Nx.tensor(0, type: Nx.type(fps)), {Nx.size(fps) + 1})
    thresholds = Nx.broadcast(thresholds_unpadded[[0]] + 1, {Nx.size(thresholds_unpadded) + 1})

    tpr = Nx.put_slice(tpr, [1], tps)
    fpr = Nx.put_slice(fpr, [1], fps)
    thresholds = Nx.put_slice(thresholds, [1], thresholds_unpadded)

    tpr = tpr / tpr[[-1]]
    fpr = fpr / fpr[[-1]]

    {fpr, tpr, thresholds}
  end

  @doc ~S"""
  Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

  Note: this implementation is restricted to the binary classification task.

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> distinct_value_indices = Scholar.Metrics.calculate_distinct_value_indices(scores)
      iex> weights = Nx.tensor([1, 1, 2, 2])
      iex> Scholar.Metrics.roc_auc_score(y_true, scores, distinct_value_indices, weights)
      #Nx.Tensor<
        f32
        0.75
      >
  """
  defn roc_auc_score(y_true, y_score, distinct_value_indices, weights) do
    {fpr, tpr, _} = roc_curve(y_true, y_score, distinct_value_indices, weights)
    auc(fpr, tpr)
  end

  @doc ~S"""
  Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

  This is equivalent to calling `Nx.roc_auc_score/4` with weights set to ones.
  """
  defn roc_auc_score(y_true, y_score, distinct_value_indices) do
    {fpr, tpr, _} = roc_curve(y_true, y_score, distinct_value_indices)
    auc(fpr, tpr)
  end

  deftransformp check_num_classes(num_classes) do
    num_classes || raise ArgumentError, "missing option :num_classes"
  end

  defnp safe_division(nominator, denominator) do
    is_zero? = denominator == 0
    nominator = Nx.select(is_zero?, 0, nominator)
    denominator = Nx.select(is_zero?, 1, denominator)
    nominator / denominator
  end

  defnp check_shape(y_true, y_pred) do
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)
  end
end
