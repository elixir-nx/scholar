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

  import Nx.Defn

  # Standard Metrics

  @doc ~S"""
  Computes the accuracy of the given predictions
  for binary classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> Scholar.Metrics.binary_accuracy(Nx.tensor([1, 0, 0]), Nx.tensor([1, 0, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn binary_accuracy(y_true, y_pred) do
    assert_shape(y_true, Nx.shape(y_pred))

    y_pred
    |> Nx.greater(0.5)
    |> Nx.equal(y_true)
    |> Nx.mean()
  end

  @doc ~S"""
  Computes the categorical accuracy of the given predictions
  for binary and multi-class classification problems
  when using one-hot encoding.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> Scholar.Metrics.onehot_accuracy(Nx.tensor([[0, 1], [1, 0], [1, 0]]), Nx.tensor([[0, 1], [1, 0], [0, 1]]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >
      iex> y_true = Nx.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
      iex> y_pred = Nx.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
      iex> Scholar.Metrics.onehot_accuracy(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.5
      >

  """
  defn onehot_accuracy(y_true, y_pred) do
    assert_shape(y_true, Nx.shape(y_pred))

    y_true
    |> Nx.argmax(axis: -1)
    |> Nx.equal(Nx.argmax(y_pred, axis: -1))
    |> Nx.mean()
  end

  @doc ~S"""
  Computes the accuracy of the given predictions
  for multi-class classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.accuracy(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 1.0, 0.25]
      >

  """
  defn accuracy(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes])
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    cm = confusion_matrix(y_true, y_pred, opts)

    true_positive = Nx.take_diagonal(cm)
    false_positive = Nx.subtract(Nx.sum(cm, axes: [0]), true_positive)

    Nx.divide(true_positive, true_positive + false_positive + 1.0e-16)
  end

  @doc ~S"""
  Computes the precision of the given predictions with respect to
  the given targets for binary classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:threshold` - threshold for truth value of the predictions.
      Defaults to `0.5`

  ## Examples

      iex> Scholar.Metrics.binary_precision(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn binary_precision(y_true, y_pred, opts \\ []) do
    assert_shape(y_true, Nx.shape(y_pred))

    true_positives = binary_true_positives(y_true, y_pred, opts)
    false_positives = binary_false_positives(y_true, y_pred, opts)

    true_positives
    |> Nx.divide(true_positives + false_positives + 1.0e-16)
  end

  @doc ~S"""
  Computes the precision of the given predictions with respect to
  the given targets for multi-class classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:threshold` - threshold for truth value of the predictions.
      Defaults to `0.5`

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
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positive = Nx.take_diagonal(cm)
    false_positive = Nx.subtract(Nx.sum(cm, axes: [0]), true_positive)
    Nx.divide(true_positive, true_positive + false_positive + 1.0e-16)
  end

  @doc ~S"""
  Computes the recall of the given predictions with respect to
  the given targets for binary classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:threshold` - threshold for truth value of the predictions.
      Defaults to `0.5`

  ## Examples

      iex> Scholar.Metrics.binary_recall(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn binary_recall(y_true, y_pred, opts \\ []) do
    assert_shape(y_true, Nx.shape(y_pred))

    true_positives = binary_true_positives(y_true, y_pred, opts)
    false_negatives = binary_false_negatives(y_true, y_pred, opts)

    Nx.divide(true_positives, false_negatives + true_positives + 1.0e-16)
  end

  @doc ~S"""
  Computes the recall of the given predictions with respect to
  the given targets for multi-class classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:threshold` - threshold for truth value of the predictions.
      Defaults to `0.5`

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
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positive = Nx.take_diagonal(cm)
    false_negative = Nx.subtract(Nx.sum(cm, axes: [1]), true_positive)
    Nx.divide(true_positive, true_positive + false_negative + 1.0e-16)
  end

  @doc """
  Computes the number of true positive predictions with respect
  to given targets for multi-class classification problems.

  ## Options

    * `:threshold` - threshold for truth value of predictions.
      Defaults to `0.5`.


  ## Examples

      iex> y_true = Nx.tensor([1, 0, 1, 1, 0, 1, 0])
      iex> y_pred = Nx.tensor([0.8, 0.6, 0.4, 0.2, 0.8, 0.2, 0.2])
      iex> Scholar.Metrics.binary_true_positives(y_true, y_pred)
      #Nx.Tensor<
        u64
        1
      >
  """
  defn binary_true_positives(y_true, y_pred, opts \\ []) do
    assert_shape(y_true, Nx.shape(y_pred))

    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds =
      y_pred
      |> Nx.greater(opts[:threshold])

    thresholded_preds
    |> Nx.equal(y_true)
    |> Nx.logical_and(Nx.equal(thresholded_preds, 1))
    |> Nx.sum()
  end

  @doc """
  Computes the number of true positive predictions with respect
  to given targets for multi-class classification problems.

  ## Options

    * `:num_classes` - Number of classes contained in the input tensors

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.true_positives(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        s64[3]
        [2, 3, 1]
      >
  """
  defn true_positives(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes])
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    confusion_matrix(y_true, y_pred, opts)
    |> Nx.take_diagonal()
  end

  @doc """
  Computes the number of false negative predictions with respect
  to given targets for binary classification problems.

  ## Options

    * `:threshold` - threshold for truth value of predictions.
      Defaults to `0.5`.

  ## Examples

      iex> y_true = Nx.tensor([1, 0, 1, 1, 0, 1, 0])
      iex> y_pred = Nx.tensor([0.8, 0.6, 0.4, 0.2, 0.8, 0.2, 0.2])
      iex> Scholar.Metrics.binary_false_negatives(y_true, y_pred)
      #Nx.Tensor<
        u64
        3
      >
  """
  defn binary_false_negatives(y_true, y_pred, opts \\ []) do
    assert_shape(y_true, Nx.shape(y_pred))

    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds =
      y_pred
      |> Nx.greater(opts[:threshold])

    thresholded_preds
    |> Nx.not_equal(y_true)
    |> Nx.logical_and(Nx.equal(thresholded_preds, 0))
    |> Nx.sum()
  end

  @doc """
  Computes the number of false negative predictions with respect
  to given targets for multi-class classification problems.

  ## Options

    * `:num_classes` - Number of classes contained in the input tensors

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.false_negatives(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        s64[3]
        [1, 3, 0]
      >
  """
  defn false_negatives(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes])
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    cm = confusion_matrix(y_true, y_pred, opts)
    Nx.subtract(Nx.sum(cm, axes: [1]), Nx.take_diagonal(cm))
  end

  @doc """
  Computes the number of true negative predictions with respect
  to given targets for binary classification problems.

  ## Options

    * `:threshold` - threshold for truth value of predictions.
      Defaults to `0.5`.

  ## Examples

      iex> y_true = Nx.tensor([1, 0, 1, 1, 0, 1, 0])
      iex> y_pred = Nx.tensor([0.8, 0.6, 0.4, 0.2, 0.8, 0.2, 0.2])
      iex> Scholar.Metrics.binary_true_negatives(y_true, y_pred)
      #Nx.Tensor<
        u64
        1
      >
  """
  defn binary_true_negatives(y_true, y_pred, opts \\ []) do
    assert_shape(y_true, Nx.shape(y_pred))

    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds =
      y_pred
      |> Nx.greater(opts[:threshold])

    thresholded_preds
    |> Nx.equal(y_true)
    |> Nx.logical_and(Nx.equal(thresholded_preds, 0))
    |> Nx.sum()
  end

  @doc """
  Computes the number of true negative predictions with respect
  to given targets for multi-class classification problems.

  ## Options

    * `:num_classes` - Number of classes contained in the input tensors

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.true_negatives(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        s64[3]
        [6, 4, 6]
      >
  """
  defn true_negatives(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes])
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positive = Nx.take_diagonal(cm)
    false_positive = Nx.subtract(Nx.sum(cm, axes: [0]), true_positive)
    false_negative = Nx.subtract(Nx.sum(cm, axes: [1]), true_positive)
    Nx.subtract(Nx.sum(cm), false_negative + false_positive + true_positive)
  end

  @doc """
  Computes the number of false positive predictions with respect
  to given targets for binary classification problems.

  ## Options

    * `:threshold` - threshold for truth value of predictions.
      Defaults to `0.5`.

  ## Examples

      iex> y_true = Nx.tensor([1, 0, 1, 1, 0, 1, 0])
      iex> y_pred = Nx.tensor([0.8, 0.6, 0.4, 0.2, 0.8, 0.2, 0.2])
      iex> Scholar.Metrics.binary_false_positives(y_true, y_pred)
      #Nx.Tensor<
        u64
        2
      >
  """
  defn binary_false_positives(y_true, y_pred, opts \\ []) do
    assert_shape(y_true, Nx.shape(y_pred))

    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds =
      y_pred
      |> Nx.greater(opts[:threshold])

    thresholded_preds
    |> Nx.not_equal(y_true)
    |> Nx.logical_and(Nx.equal(thresholded_preds, 1))
    |> Nx.sum()
  end

  @doc """
  Computes the number of false positives predictions with respect
  to given targets for multi-class classification problems.

  ## Options

    * `:num_classes` - Number of classes contained in the input tensors

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.false_positives(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        s64[3]
        [1, 0, 3]
      >
  """
  defn false_positives(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, [:num_classes])
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    cm = confusion_matrix(y_true, y_pred, opts)
    Nx.subtract(Nx.sum(cm, axes: [0]), Nx.take_diagonal(cm))
  end

  @doc ~S"""
  Computes the sensitivity of the given predictions with respect
  to the given targets for binary classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:threshold` - threshold for truth value of the predictions.
      Defaults to `0.5`

  ## Examples

      iex> Scholar.Metrics.binary_sensitivity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn binary_sensitivity(y_true, y_pred, opts \\ []) do
    assert_shape(y_true, Nx.shape(y_pred))

    opts = keyword!(opts, threshold: 0.5)

    binary_recall(y_true, y_pred, opts)
  end

  @doc ~S"""
  Computes the sensitivity of the given predictions with respect
  to the given targets for multi-class classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

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
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    recall(y_true, y_pred, opts)
  end

  @doc ~S"""
  Computes the specificity of the given predictions with respect
  to the given targets for binary classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:threshold` - threshold for truth value of the predictions.
      Defaults to `0.5`

  ## Examples

      iex> Scholar.Metrics.binary_specificity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.0
      >

  """
  defn binary_specificity(y_true, y_pred, opts \\ []) do
    assert_shape(y_true, Nx.shape(y_pred))

    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds = Nx.greater(y_pred, opts[:threshold])

    true_negatives =
      thresholded_preds
      |> Nx.equal(y_true)
      |> Nx.logical_and(Nx.equal(thresholded_preds, 0))
      |> Nx.sum()

    false_positives =
      thresholded_preds
      |> Nx.not_equal(y_true)
      |> Nx.logical_and(Nx.equal(thresholded_preds, 1))
      |> Nx.sum()

    Nx.divide(true_negatives, false_positives + true_negatives + 1.0e-16)
  end

  @doc ~S"""
  Computes the specificity of the given predictions with respect
  to the given targets for binary classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

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
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    true_negatives = true_negatives(y_true, y_pred, opts)
    false_positives = false_positives(y_true, y_pred, opts)

    Nx.divide(true_negatives, false_positives + true_negatives + 1.0e-16)
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
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    num_classes =
      transform(opts[:num_classes], fn num_classes ->
        num_classes || raise ArgumentError, "missing option :num_classes"
      end)

    zeros = Nx.broadcast(0, {num_classes, num_classes})
    indices = Nx.stack([y_true, y_pred], axis: 1)
    updates = Nx.broadcast(1, {Nx.size(y_true)})

    Nx.indexed_add(zeros, indices, updates)
  end

  @doc ~S"""
  Calculates the mean absolute error of predictions
  with respect to targets.

  $$l_i = \sum_i |\hat{y_i} - y_i|$$

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

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
    assert_shape(y_true, Nx.shape(y_pred))

    y_true
    |> Nx.subtract(y_pred)
    |> Nx.abs()
    |> Nx.mean()
  end

  # Combinators

  @doc """
  Returns a function which computes a running average given current average,
  new observation, and current iteration.

  ## Examples

      iex> cur_avg = 0.5
      iex> iteration = 1
      iex> y_true = Nx.tensor([0, 1, 1])
      iex> y_pred = Nx.tensor([0, 1, 1])
      iex> avg_acc = Scholar.Metrics.running_average(&Scholar.Metrics.binary_accuracy/2)
      iex> avg_acc.(cur_avg, [y_true, y_pred], iteration)
      #Nx.Tensor<
        f32
        0.75
      >
      iex> y_true = Nx.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
      iex> y_pred = Nx.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
      iex> avg_acc = Scholar.Metrics.running_average(&Scholar.Metrics.onehot_accuracy/2)
      iex> avg_acc.(cur_avg, [y_true, y_pred], iteration)
      #Nx.Tensor<
        f32
        0.75
      >
      iex> y_true = Nx.tensor([0, 1, 2])
      iex> y_pred = Nx.tensor([0, 1, 2])
      iex> avg_acc = Scholar.Metrics.running_average(&Scholar.Metrics.accuracy(&1, &2, num_classes: 3))
      iex> avg_acc.(cur_avg, [y_true, y_pred], iteration)
      #Nx.Tensor<
        f32[3]
        [0.75, 0.75, 0.75]
      >
  """
  def running_average(metric) do
    &running_average_impl(&1, apply(metric, &2), &3)
  end

  defnp running_average_impl(avg, obs, i) do
    avg
    |> Nx.multiply(i)
    |> Nx.add(obs)
    |> Nx.divide(Nx.add(i, 1))
  end

  @doc """
  Returns a function which computes a running sum given current sum,
  new observation, and current iteration.

  ## Examples

      iex> cur_sum = 12
      iex> iteration = 2
      iex> y_true = Nx.tensor([0, 1, 0, 1])
      iex> y_pred = Nx.tensor([1, 1, 0, 1])
      iex> fps = Scholar.Metrics.running_sum(&Scholar.Metrics.binary_false_positives/2)
      iex> fps.(cur_sum, [y_true, y_pred], iteration)
      #Nx.Tensor<
        s64
        13
      >
      iex> cur_sum = Nx.tensor([10, 20, 30])
      iex> iteration = 2
      iex> y_true = Nx.tensor([0, 1, 2, 1])
      iex> y_pred = Nx.tensor([1, 1, 2, 0])
      iex> fps = Scholar.Metrics.running_sum(&Scholar.Metrics.false_positives(&1, &2, num_classes: 3))
      iex> fps.(cur_sum, [y_true, y_pred], iteration)
      #Nx.Tensor<
        s64[3]
        [11, 21, 30]
      >
  """
  def running_sum(metric) do
    &running_sum_impl(&1, apply(metric, &2), &3)
  end

  defnp running_sum_impl(sum, obs, _) do
    Nx.add(sum, obs)
  end
end
