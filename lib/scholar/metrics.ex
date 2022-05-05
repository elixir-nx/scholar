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
  for binary and multi-class classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

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
    assert_shape(y_pred, Nx.shape(y_true))

    y_pred
    |> Nx.equal(y_true)
    |> Nx.mean()
  end

  @doc ~S"""
  Computes the precision of the given predictions with respect to
  the given targets for binary classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> Scholar.Metrics.binary_precision(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn binary_precision(y_true, y_pred) do
    assert_shape(y_true, Nx.shape(y_pred))

    true_positives = binary_true_positives(y_true, y_pred)
    false_positives = binary_false_positives(y_true, y_pred)

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
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positives = Nx.take_diagonal(cm)
    false_positives = Nx.subtract(Nx.sum(cm, axes: [0]), true_positives)

    true_positives
    |> Nx.divide(true_positives + false_positives + 1.0e-16)
  end

  @doc ~S"""
  Computes the recall of the given predictions with respect to
  the given targets for binary classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> Scholar.Metrics.binary_recall(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn binary_recall(y_true, y_pred) do
    assert_shape(y_true, Nx.shape(y_pred))

    true_positives = binary_true_positives(y_true, y_pred)
    false_negatives = binary_false_negatives(y_true, y_pred)

    Nx.divide(true_positives, false_negatives + true_positives + 1.0e-16)
  end

  @doc ~S"""
  Computes the recall of the given predictions with respect to
  the given targets for multi-class classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

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
    assert_shape_pattern(y_true, {_})
    assert_shape(y_pred, Nx.shape(y_true))

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positive = Nx.take_diagonal(cm)
    false_negative = Nx.subtract(Nx.sum(cm, axes: [1]), true_positive)
    Nx.divide(true_positive, true_positive + false_negative + 1.0e-16)
  end

  defnp binary_true_positives(y_true, y_pred) do
    assert_shape(y_true, Nx.shape(y_pred))

    y_pred
    |> Nx.equal(y_true)
    |> Nx.logical_and(Nx.equal(y_pred, 1))
    |> Nx.sum()
  end

  defnp binary_false_negatives(y_true, y_pred) do
    assert_shape(y_true, Nx.shape(y_pred))

    y_pred
    |> Nx.not_equal(y_true)
    |> Nx.logical_and(Nx.equal(y_pred, 0))
    |> Nx.sum()
  end

  defnp binary_true_negatives(y_true, y_pred) do
    assert_shape(y_true, Nx.shape(y_pred))

    y_pred
    |> Nx.equal(y_true)
    |> Nx.logical_and(Nx.equal(y_pred, 0))
    |> Nx.sum()
  end

  defnp binary_false_positives(y_true, y_pred) do
    assert_shape(y_true, Nx.shape(y_pred))

    y_pred
    |> Nx.not_equal(y_true)
    |> Nx.logical_and(Nx.equal(y_pred, 1))
    |> Nx.sum()
  end

  @doc ~S"""
  Computes the sensitivity of the given predictions with respect
  to the given targets for binary classification problems.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> Scholar.Metrics.binary_sensitivity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn binary_sensitivity(y_true, y_pred) do
    assert_shape(y_true, Nx.shape(y_pred))

    binary_recall(y_true, y_pred)
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

  ## Examples

      iex> Scholar.Metrics.binary_specificity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.0
      >

  """
  defn binary_specificity(y_true, y_pred) do
    assert_shape(y_true, Nx.shape(y_pred))

    true_negatives =
      y_pred
      |> Nx.equal(y_true)
      |> Nx.logical_and(Nx.equal(y_pred, 0))
      |> Nx.sum()

    false_positives =
      y_pred
      |> Nx.not_equal(y_true)
      |> Nx.logical_and(Nx.equal(y_pred, 1))
      |> Nx.sum()

    Nx.divide(true_negatives, false_positives + true_negatives + 1.0e-16)
  end

  @doc ~S"""
  Computes the specificity of the given predictions with respect
  to the given targets for multi-class classification problems.

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

    cm = confusion_matrix(y_true, y_pred, opts)
    true_positive = Nx.take_diagonal(cm)
    false_positive = Nx.subtract(Nx.sum(cm, axes: [0]), true_positive)
    false_negative = Nx.subtract(Nx.sum(cm, axes: [1]), true_positive)
    true_negative = Nx.subtract(Nx.sum(cm), false_negative + false_positive + true_positive)

    Nx.divide(true_negative, false_positive + true_negative + 1.0e-16)
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
      iex> avg_acc = Scholar.Metrics.running_average(&Scholar.Metrics.accuracy/2)
      iex> avg_acc.(cur_avg, [y_true, y_pred], iteration)
      #Nx.Tensor<
        f32
        0.75
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
  """
  def running_sum(metric) do
    &running_sum_impl(&1, apply(metric, &2), &3)
  end

  defnp running_sum_impl(sum, obs, _) do
    Nx.add(sum, obs)
  end
end
