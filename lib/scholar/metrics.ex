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
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)

    (y_pred == y_true)
    |> Nx.mean()
  end

  @doc ~S"""
  Computes the precision of the given predictions with respect to
  the given targets for binary classification problems.

  ## Examples

      iex> Scholar.Metrics.binary_precision(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn binary_precision(y_true, y_pred) do
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)

    true_positives = binary_true_positives(y_true, y_pred)
    false_positives = binary_false_positives(y_true, y_pred)

    true_positives / (true_positives + false_positives + 1.0e-16)
  end

  @doc ~S"""
  Computes the precision of the given predictions with respect to
  the given targets for multi-class classification problems.

  ## Options

    * `:num_classes` - Number of classes contained in the input tensors

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.precision(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 1.0, 0.25]
      >

  """
  defn precision(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes])
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positives = Nx.take_diagonal(cm)
    false_positives = Nx.sum(cm, axes: [0]) - true_positives

    true_positives / (true_positives + false_positives + 1.0e-16)
  end

  @doc ~S"""
  Computes the recall of the given predictions with respect to
  the given targets for binary classification problems.

  ## Examples

      iex> Scholar.Metrics.binary_recall(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn binary_recall(y_true, y_pred) do
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)

    true_positives = binary_true_positives(y_true, y_pred)
    false_negatives = binary_false_negatives(y_true, y_pred)

    true_positives / (false_negatives + true_positives + 1.0e-16)
  end

  @doc ~S"""
  Computes the recall of the given predictions with respect to
  the given targets for multi-class classification problems.

  ## Options

    * `:num_classes` - Number of classes contained in the input tensors

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.recall(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 0.5, 1.0]
      >

  """
  defn recall(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes])
    assert_rank!(y_true, 1)
    assert_same_shape!(y_pred, y_true)

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positive = Nx.take_diagonal(cm)
    false_negative = Nx.sum(cm, axes: [1]) - true_positive
    true_positive / (true_positive + false_negative + 1.0e-16)
  end

  defnp binary_true_positives(y_true, y_pred) do
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)

    (y_pred == y_true and y_pred == 1)
    |> Nx.sum()
  end

  defnp binary_false_negatives(y_true, y_pred) do
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)

    (y_pred != y_true and y_pred == 0)
    |> Nx.sum()
  end

  defnp binary_true_negatives(y_true, y_pred) do
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)

    (y_pred == y_true and y_pred == 0)
    |> Nx.sum()
  end

  defnp binary_false_positives(y_true, y_pred) do
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)

    (y_pred != y_true and y_pred == 1)
    |> Nx.sum()
  end

  @doc ~S"""
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
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)

    binary_recall(y_true, y_pred)
  end

  @doc ~S"""
  Computes the sensitivity of the given predictions with respect
  to the given targets for multi-class classification problems.

  ## Options

    * `:num_classes` - Number of classes contained in the input tensors

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.sensitivity(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 0.5, 1.0]
      >

  """
  defn sensitivity(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes])
    assert_rank!(y_true, 1)
    assert_same_shape!(y_pred, y_true)

    recall(y_true, y_pred, opts)
  end

  @doc ~S"""
  Computes the specificity of the given predictions with respect
  to the given targets for binary classification problems.

  ## Examples

      iex> Scholar.Metrics.binary_specificity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.0
      >

  """
  defn binary_specificity(y_true, y_pred) do
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)

    true_negatives = binary_true_negatives(y_true, y_pred)
    false_positives = binary_false_positives(y_true, y_pred)

    true_negatives / (false_positives + true_negatives + 1.0e-16)
  end

  @doc ~S"""
  Computes the specificity of the given predictions with respect
  to the given targets for multi-class classification problems.

  ## Options

    * `:num_classes` - Number of classes contained in the input tensors

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.specificity(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.8571428656578064, 1.0, 0.6666666865348816]
      >

  """
  defn specificity(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes])
    assert_rank!(y_true, 1)
    assert_same_shape!(y_pred, y_true)

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positive = Nx.take_diagonal(cm)
    false_positive = Nx.sum(cm, axes: [0]) - true_positive
    false_negative = Nx.sum(cm, axes: [1]) - true_positive
    true_negative = Nx.sum(cm) - (false_negative + false_positive + true_positive)

    true_negative / (false_positive + true_negative + 1.0e-16)
  end

  @doc ~S"""
  Calculates the confusion matrix given rank-1 tensors which represent
  the expected (`y_true`) and predicted (`y_pred`) classes.

  ## Options

    * `:num_classes` - required. Number of classes contained in the input tensors

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1, 2, 2], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 1, 0, 2, 2, 2], type: {:u, 32})
      iex> Scholar.Metrics.confusion_matrix(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        s64[3][3]
        [
          [1, 1, 0],
          [1, 0, 1],
          [0, 0, 2]
        ]
      >

  """
  defn confusion_matrix(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes])
    assert_rank!(y_true, 1)
    assert_same_shape!(y_pred, y_true)

    num_classes = check_num_classes(opts[:num_classes])

    zeros = Nx.broadcast(0, {num_classes, num_classes})
    indices = Nx.stack([y_true, y_pred], axis: 1)
    updates = Nx.broadcast(1, {Nx.size(y_true)})

    Nx.indexed_add(zeros, indices, updates)
  end

  @doc ~S"""
  Calculates F1 score given rank-1 tensors which represent
  the expected (`y_true`) and predicted (`y_pred`) classes.

  ## Options

    * `:num_classes` - required. Number of classes contained in the input tensors
    * `:average` - optional. This determines the type of averaging performed on the data.
      * `:macro`. Calculate metrics for each label, and find their unweighted mean.
        This does not take label imbalance into account.
      * `:weighted`. Calculate metrics for each label, and find their average weighted by
        support (the number of true instances for each label).
      * `:micro`. Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
      * If not specified, then the f1 scores for each class are returned.

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.f1_score(y_true, y_pred, num_classes: 3, average: nil)
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
  """
  defn f1_score(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes, :average])

    assert_rank!(y_true, 1)
    assert_same_shape!(y_pred, y_true)

    num_classes = check_num_classes(opts[:num_classes])

    case opts[:average] do
      :micro ->
        accuracy(y_true, y_pred)

      _ ->
        cm = confusion_matrix(y_true, y_pred, num_classes: num_classes)
        true_positive = Nx.take_diagonal(cm)
        false_positive = Nx.sum(cm, axes: [0]) - true_positive
        false_negative = Nx.sum(cm, axes: [1]) - true_positive

        precision = true_positive / (true_positive + false_positive + 1.0e-16)

        recall = true_positive / (true_positive + false_negative + 1.0e-16)

        per_class_f1 = 2 * precision * recall / (precision + recall + 1.0e-16)

        case opts[:average] do
          nil ->
            per_class_f1

          :macro ->
            Nx.mean(per_class_f1)

          :weighted ->
            support = (y_true == Nx.iota({num_classes, 1})) |> Nx.sum(axes: [1])

            (per_class_f1 * support / (Nx.sum(support) + 1.0e-16))
            |> Nx.sum()
        end
    end
  end

  @doc ~S"""
  Calculates the mean absolute error of predictions
  with respect to targets.

  $$l_i = \sum_i |\hat{y_i} - y_i|$$

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

  deftransformp check_num_classes(num_classes) do
    num_classes || raise ArgumentError, "missing option :num_classes"
  end
end
