defmodule Scholar.Metrics.Classification do
  @moduledoc """
  Classification Metric functions.

  Metrics are used to measure the performance and compare
  the performance of any kind of classifier in
  easy-to-understand terms.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  import Nx.Defn, except: [assert_shape: 2, assert_shape_pattern: 2]
  import Scholar.Shared
  import Scholar.Preprocessing
  alias Scholar.Integrate

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

          * `:none` - The F-score values for each class are returned.
          """
        ]
      ]

  fbeta_score_schema = f1_score_schema

  precision_recall_fscore_support_schema =
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

          * `:none` - The F-score values for each class are returned.
          """
        ],
        beta: [
          type: {:custom, Scholar.Options, :beta, []},
          default: 1,
          doc: """
          Determines the weight of recall in the combined score.
          For values of `beta` > 1 it gives more weight to recall, while `beta` < 1 favors precision.
          """
        ]
      ]

  confusion_matrix_schema =
    general_schema ++
      [
        sample_weights: [
          type: {:custom, Scholar.Options, :weights, []},
          doc: """
          Sample weights of the observations.
          """
        ],
        normalize: [
          type: {:in, [true, :predicted, :all]},
          doc: """
          Normalizes confusion matrix over the `:true` (rows), `:predicted` (columns)
          conditions or `:all` the population. If `nil`, confusion matrix will not be normalized.
          """
        ]
      ]

  brier_score_loss_schema = [
    sample_weights: [
      type: {:custom, Scholar.Options, :weights, []},
      default: 1.0,
      doc: """
      Sample weights of the observations.
      """
    ],
    pos_label: [
      type: :integer,
      default: 1,
      doc: """
      Label of the positive class.
      """
    ]
  ]

  balanced_accuracy_schema =
    general_schema ++
      [
        sample_weights: [
          type: {:custom, Scholar.Options, :weights, []},
          doc: """
          Sample weights of the observations.
          """
        ],
        adjusted: [
          type: :boolean,
          default: false,
          doc: """
          If `true`, the balanced accuracy is adjusted for chance
          (depends on the number of classes).
          """
        ]
      ]

  cohen_kappa_schema =
    general_schema ++
      [
        weights: [
          type: {:custom, Scholar.Options, :weights, []},
          doc: """
          Weighting to calculate the score.
          """
        ],
        weighting_type: [
          type: {:in, [:linear, :quadratic]},
          doc: """
          Weighting type to calculate the score.
          """
        ]
      ]

  accuracy_schema = [
    normalize: [
      type: :boolean,
      default: true,
      doc: """
      If `true`, return the fraction of correctly classified samples.
      Otherwise, return the number of correctly classified samples.
      """
    ]
  ]

  log_loss_schema =
    general_schema ++
      [
        normalize: [
          type: :boolean,
          default: true,
          doc: """
          If `true`, return the mean loss over the samples.
          Otherwise, return the sum of losses over the samples.
          """
        ],
        sample_weights: [
          type: {:custom, Scholar.Options, :weights, []},
          default: 1.0,
          doc: """
          Sample weights of the observations.
          """
        ]
      ]

  top_k_accuracy_score_schema =
    general_schema ++
      [
        k: [
          type: :integer,
          default: 5,
          doc: """
          Number of top elements to look at for computing accuracy.
          """
        ],
        normalize: [
          type: :boolean,
          default: true,
          doc: """
          If `true`, return the fraction of correctly classified samples.
          Otherwise, return the number of correctly classified samples.
          """
        ]
      ]

  zero_one_loss_schema = [
    normalize: [
      type: :boolean,
      default: true,
      doc: """
      If `true`, return the fraction of incorrectly classified samples.
      Otherwise, return the number of incorrectly classified samples.
      """
    ]
  ]

  @general_schema NimbleOptions.new!(general_schema)
  @confusion_matrix_schema NimbleOptions.new!(confusion_matrix_schema)
  @balanced_accuracy_schema NimbleOptions.new!(balanced_accuracy_schema)
  @cohen_kappa_schema NimbleOptions.new!(cohen_kappa_schema)
  @fbeta_score_schema NimbleOptions.new!(fbeta_score_schema)
  @f1_score_schema NimbleOptions.new!(f1_score_schema)
  @precision_recall_fscore_support_schema NimbleOptions.new!(
                                            precision_recall_fscore_support_schema
                                          )
  @brier_score_loss_schema NimbleOptions.new!(brier_score_loss_schema)
  @accuracy_schema NimbleOptions.new!(accuracy_schema)
  @log_loss_schema NimbleOptions.new!(log_loss_schema)
  @top_k_accuracy_score_schema NimbleOptions.new!(top_k_accuracy_score_schema)
  @zero_one_loss_schema NimbleOptions.new!(zero_one_loss_schema)

  # Standard Metrics

  @doc ~S"""
  Computes the accuracy of the given predictions
  for binary and multi-class classification problems.

  ## Examples

      iex> Scholar.Metrics.Classification.accuracy(Nx.tensor([1, 0, 0]), Nx.tensor([1, 0, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: :u32)
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: :u32)
      iex> Scholar.Metrics.Classification.accuracy(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.6000000238418579
      >

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: :u32)
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: :u32)
      iex> Scholar.Metrics.Classification.accuracy(y_true, y_pred, normalize: false)
      #Nx.Tensor<
        u64
        6
      >
  """
  deftransform accuracy(y_true, y_pred, opts \\ []) do
    accuracy_n(y_true, y_pred, NimbleOptions.validate!(opts, @accuracy_schema))
  end

  defnp accuracy_n(y_true, y_pred, opts) do
    check_shape(y_true, y_pred)

    case opts[:normalize] do
      true ->
        Nx.mean(y_pred == y_true)

      false ->
        Nx.sum(y_pred == y_true)
    end
  end

  @doc ~S"""
  Computes the precision of the given predictions with respect to
  the given targets for binary classification problems.

  If the sum of true positives and false positives is 0, then the
  result is 0 to avoid zero division.

  ## Examples

      iex> Scholar.Metrics.Classification.binary_precision(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
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

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: :u32)
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: :u32)
      iex> Scholar.Metrics.Classification.precision(y_true, y_pred, num_classes: 3)
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

      iex> Scholar.Metrics.Classification.binary_recall(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
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

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: :u32)
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: :u32)
      iex> Scholar.Metrics.Classification.recall(y_true, y_pred, num_classes: 3)
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

      iex> Scholar.Metrics.Classification.binary_sensitivity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
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

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: :u32)
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: :u32)
      iex> Scholar.Metrics.Classification.sensitivity(y_true, y_pred, num_classes: 3)
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

      iex> Scholar.Metrics.Classification.binary_specificity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
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

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: :u32)
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: :u32)
      iex> Scholar.Metrics.Classification.specificity(y_true, y_pred, num_classes: 3)
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

      iex> y_true = Nx.tensor([0, 0, 1, 1, 2, 2], type: :u32)
      iex> y_pred = Nx.tensor([0, 1, 0, 2, 2, 2], type: :u32)
      iex> Scholar.Metrics.Classification.confusion_matrix(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        u64[3][3]
        [
          [1, 1, 0],
          [1, 0, 1],
          [0, 0, 2]
        ]
      >

      iex> y_true = Nx.tensor([0, 0, 1, 1, 2, 2], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 1, 0, 2, 2, 2], type: {:u, 32})
      iex> sample_weights = [2, 5, 1, 1.5, 2, 8]
      iex> Scholar.Metrics.Classification.confusion_matrix(y_true, y_pred, num_classes: 3, sample_weights: sample_weights, normalize: :predicted)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.6666666865348816, 1.0, 0.0],
          [0.3333333432674408, 0.0, 0.1304347813129425],
          [0.0, 0.0, 0.8695651888847351]
        ]
      >
  """
  deftransform confusion_matrix(y_true, y_pred, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @confusion_matrix_schema)

    weights =
      if opts[:sample_weights] == nil,
        do: Nx.u64(1),
        else: validate_weights(opts[:sample_weights], Nx.axis_size(y_true, 0))

    confusion_matrix_n(y_true, y_pred, weights, opts)
  end

  defnp confusion_matrix_n(y_true, y_pred, weights, opts) do
    check_shape(y_pred, y_true)

    num_classes = check_num_classes(opts[:num_classes])

    zeros = Nx.broadcast(Nx.u64(0), {num_classes, num_classes})
    indices = Nx.stack([y_true, y_pred], axis: 1)
    updates = Nx.broadcast(Nx.u64(1), y_true) * weights

    cm = Nx.indexed_add(zeros, indices, updates)

    case opts[:normalize] do
      true ->
        cm / Nx.sum(cm, axes: [1], keep_axes: true)

      :predicted ->
        cm / Nx.sum(cm, axes: [0], keep_axes: true)

      :all ->
        cm / Nx.sum(cm)

      _ ->
        cm
    end
  end

  @doc """
  Computes the balanced accuracy score for multi-class classification

  ## Options

  #{NimbleOptions.docs(@balanced_accuracy_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 2, 0, 1, 2], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 0, 0, 1], type: {:u, 32})
      iex> Scholar.Metrics.Classification.balanced_accuracy_score(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32
        0.3333333432674408
      >
      iex> y_true = Nx.tensor([0, 1, 2, 0, 1, 2], type: {:u, 32})
      iex> y_pred = Nx.tensor([0, 2, 1, 0, 0, 1], type: {:u, 32})
      iex> sample_weights = [1, 1, 1, 2, 2, 2]
      iex> Scholar.Metrics.Classification.balanced_accuracy_score(y_true, y_pred, num_classes: 3, sample_weights: sample_weights, adjusted: true)
      #Nx.Tensor<
        f32
        0.0
      >
  """
  deftransform balanced_accuracy_score(y_true, y_pred, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @balanced_accuracy_schema)

    balanced_accuracy_score_n(y_true, y_pred, opts)
  end

  defnp balanced_accuracy_score_n(y_true, y_pred, opts) do
    check_shape(y_pred, y_true)

    cm =
      confusion_matrix(y_true, y_pred,
        sample_weights: opts[:sample_weights],
        num_classes: opts[:num_classes]
      )

    per_class = Nx.take_diagonal(cm)
    sums = Nx.sum(cm, axes: [1])
    num_zeros = Nx.sum(sums == 0)
    per_class = per_class / Nx.select(sums == 0, Nx.f32(1), sums)
    score = Nx.sum(per_class) / (opts[:num_classes] - num_zeros)

    if opts[:adjusted] do
      num_classes = opts[:num_classes] - num_zeros
      chance = 1 / num_classes
      (score - chance) / (1 - chance)
    else
      score
    end
  end

  @doc """
  Calculates F-beta score given rank-1 tensors which represent
  the expected (`y_true`) and predicted (`y_pred`) classes.

  If all examples are true negatives, then the result is 0 to
  avoid zero division.

  #{~S'''
  $$F_\beta = (1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{(\beta^2 \cdot \mathrm{precision}) + \mathrm{recall}}$$
  '''}

  ## Options

  #{NimbleOptions.docs(@fbeta_score_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: :u32)
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: :u32)
      iex> Scholar.Metrics.Classification.fbeta_score(y_true, y_pred, Nx.u32(1), num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 0.6666666865348816, 0.4000000059604645]
      >
      iex> Scholar.Metrics.Classification.fbeta_score(y_true, y_pred, Nx.u32(2), num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 0.5555555820465088, 0.625]
      >
      iex> Scholar.Metrics.Classification.fbeta_score(y_true, y_pred, Nx.f32(0.5), num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 0.8333333134651184, 0.29411765933036804]
      >
      iex> Scholar.Metrics.Classification.fbeta_score(y_true, y_pred, Nx.u32(2), num_classes: 3, average: :macro)
      #Nx.Tensor<
        f32
        0.6157407760620117
      >
      iex> Scholar.Metrics.Classification.fbeta_score(y_true, y_pred, Nx.u32(2), num_classes: 3, average: :weighted)
      #Nx.Tensor<
        f32
        0.5958333611488342
      >
      iex> Scholar.Metrics.Classification.fbeta_score(y_true, y_pred, Nx.f32(0.5), num_classes: 3, average: :micro)
      #Nx.Tensor<
        f32
        0.6000000238418579
      >
      iex> Scholar.Metrics.Classification.fbeta_score(Nx.tensor([1, 0, 1, 0]), Nx.tensor([0, 1, 0, 1]), Nx.tensor(0.5), num_classes: 2, average: :none)
      #Nx.Tensor<
        f32[2]
        [0.0, 0.0]
      >
      iex> Scholar.Metrics.Classification.fbeta_score(Nx.tensor([1, 0, 1, 0]), Nx.tensor([0, 1, 0, 1]), 0.5, num_classes: 2, average: :none)
      #Nx.Tensor<
        f32[2]
        [0.0, 0.0]
      >
  """
  deftransform fbeta_score(y_true, y_pred, beta, opts \\ []) do
    fbeta_score_n(y_true, y_pred, beta, NimbleOptions.validate!(opts, @fbeta_score_schema))
  end

  defnp fbeta_score_n(y_true, y_pred, beta, opts) do
    {_precision, _recall, per_class_fscore, _support} =
      precision_recall_fscore_support_n(y_true, y_pred, beta, opts)

    per_class_fscore
  end

  defnp fbeta_score_v(confusion_matrix, average) do
    true_positive = Nx.take_diagonal(confusion_matrix)
    false_positive = Nx.sum(confusion_matrix, axes: [0]) - true_positive
    false_negative = Nx.sum(confusion_matrix, axes: [1]) - true_positive

    case average do
      :micro ->
        true_positive = Nx.sum(true_positive)
        false_positive = Nx.sum(false_positive)
        false_negative = Nx.sum(false_negative)

        {true_positive, false_positive, false_negative}

      _ ->
        {true_positive, false_positive, false_negative}
    end
  end

  @doc """
  Calculates precision, recall, F-score and support for each
  class. It also supports a `beta` argument which weights
  recall more than precision by it's value.

  ## Options

  #{NimbleOptions.docs(@precision_recall_fscore_support_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: :u32)
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: :u32)
      iex> Scholar.Metrics.Classification.precision_recall_fscore_support(y_true, y_pred, num_classes: 3)
      {Nx.f32([0.6666666865348816, 1.0, 0.25]),
       Nx.f32([0.6666666865348816, 0.5, 1.0]),
       Nx.f32([0.6666666865348816, 0.6666666865348816, 0.4000000059604645]),
       Nx.u64([3, 6, 1])}
      iex> Scholar.Metrics.Classification.precision_recall_fscore_support(y_true, y_pred, num_classes: 3, average: :macro)
      {Nx.f32([0.6666666865348816, 1.0, 0.25]),
       Nx.f32([0.6666666865348816, 0.5, 1.0]),
       Nx.f32(0.5777778029441833),
       Nx.Constants.nan()}
      iex> Scholar.Metrics.Classification.precision_recall_fscore_support(y_true, y_pred, num_classes: 3, average: :weighted)
      {Nx.f32([0.6666666865348816, 1.0, 0.25]),
       Nx.f32([0.6666666865348816, 0.5, 1.0]),
       Nx.f32(0.6399999856948853),
       Nx.Constants.nan()}
      iex> Scholar.Metrics.Classification.precision_recall_fscore_support(y_true, y_pred, num_classes: 3, average: :micro)
      {Nx.f32(0.6000000238418579),
       Nx.f32(0.6000000238418579),
       Nx.f32(0.6000000238418579),
       Nx.Constants.nan()}

      iex> y_true = Nx.tensor([1, 0, 1, 0], type: :u32)
      iex> y_pred = Nx.tensor([0, 1, 0, 1], type: :u32)
      iex> opts = [beta: 2, num_classes: 2, average: :none]
      iex> Scholar.Metrics.Classification.precision_recall_fscore_support(y_true, y_pred, opts)
      {Nx.f32([0.0, 0.0]),
       Nx.f32([0.0, 0.0]),
       Nx.f32([0.0, 0.0]),
       Nx.u64([2, 2])}
  """
  deftransform precision_recall_fscore_support(y_true, y_pred, opts) do
    opts = NimbleOptions.validate!(opts, @precision_recall_fscore_support_schema)
    {beta, opts} = Keyword.pop(opts, :beta)

    precision_recall_fscore_support_n(
      y_true,
      y_pred,
      beta,
      opts
    )
  end

  defnp precision_recall_fscore_support_n(y_true, y_pred, beta, opts) do
    check_shape(y_pred, y_true)
    num_classes = check_num_classes(opts[:num_classes])
    average = opts[:average]

    confusion_matrix = confusion_matrix(y_true, y_pred, num_classes: num_classes)
    {true_positive, false_positive, false_negative} = fbeta_score_v(confusion_matrix, average)

    precision = safe_division(true_positive, true_positive + false_positive)
    recall = safe_division(true_positive, true_positive + false_negative)

    per_class_fscore =
      cond do
        # Should only be +Inf
        Nx.is_infinity(beta) ->
          recall

        beta == 0 ->
          precision

        true ->
          beta2 = Nx.pow(beta, 2)
          safe_division((1 + beta2) * precision * recall, beta2 * precision + recall)
      end

    case average do
      :none ->
        support = (y_true == Nx.iota({num_classes, 1})) |> Nx.sum(axes: [1])

        {precision, recall, per_class_fscore, support}

      :micro ->
        {precision, recall, per_class_fscore, Nx.Constants.nan()}

      :macro ->
        {precision, recall, Nx.mean(per_class_fscore), Nx.Constants.nan()}

      :weighted ->
        support = (y_true == Nx.iota({num_classes, 1})) |> Nx.sum(axes: [1])

        per_class_fscore =
          (per_class_fscore * support)
          |> safe_division(Nx.sum(support))
          |> Nx.sum()

        {precision, recall, per_class_fscore, Nx.Constants.nan()}
    end
  end

  @doc """
  Calculates F1 score given rank-1 tensors which represent
  the expected (`y_true`) and predicted (`y_pred`) classes.

  If all examples are true negatives, then the result is 0 to
  avoid zero division.

  ## Options

  #{NimbleOptions.docs(@f1_score_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 1, 1, 0, 2, 1, 0, 1], type: :u32)
      iex> y_pred = Nx.tensor([0, 2, 1, 1, 2, 2, 2, 0, 0, 1], type: :u32)
      iex> Scholar.Metrics.Classification.f1_score(y_true, y_pred, num_classes: 3)
      #Nx.Tensor<
        f32[3]
        [0.6666666865348816, 0.6666666865348816, 0.4000000059604645]
      >
      iex> Scholar.Metrics.Classification.f1_score(y_true, y_pred, num_classes: 3, average: :macro)
      #Nx.Tensor<
        f32
        0.5777778029441833
      >
      iex> Scholar.Metrics.Classification.f1_score(y_true, y_pred, num_classes: 3, average: :weighted)
      #Nx.Tensor<
        f32
        0.6399999856948853
      >
      iex> Scholar.Metrics.Classification.f1_score(y_true, y_pred, num_classes: 3, average: :micro)
      #Nx.Tensor<
        f32
        0.6000000238418579
      >
      iex> Scholar.Metrics.Classification.f1_score(Nx.tensor([1, 0, 1, 0]), Nx.tensor([0, 1, 0, 1]), num_classes: 2, average: :none)
      #Nx.Tensor<
        f32[2]
        [0.0, 0.0]
      >
  """
  deftransform f1_score(y_true, y_pred, opts \\ []) do
    fbeta_score_n(y_true, y_pred, 1, NimbleOptions.validate!(opts, @f1_score_schema))
  end

  @doc """
  Zero-one classification loss.

  ## Options

  #{NimbleOptions.docs(@zero_one_loss_schema)}

  # Examples

      iex> y_pred = Nx.tensor([1, 2, 3, 4])
      iex> y_true = Nx.tensor([2, 2, 3, 4])
      iex> Scholar.Metrics.Classification.zero_one_loss(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.25
      >

      iex> y_pred = Nx.tensor([1, 2, 3, 4])
      iex> y_true = Nx.tensor([2, 2, 3, 4])
      iex> Scholar.Metrics.Classification.zero_one_loss(y_true, y_pred, normalize: false)
      #Nx.Tensor<
        u64
        1
      >
  """
  deftransform zero_one_loss(y_true, y_pred, opts \\ []) do
    zero_one_loss_n(y_true, y_pred, NimbleOptions.validate!(opts, @zero_one_loss_schema))
  end

  defnp zero_one_loss_n(y_true, y_pred, opts) do
    case opts[:normalize] do
      true ->
        1 - accuracy(y_true, y_pred, opts)

      false ->
        Nx.axis_size(y_true, 0) - accuracy(y_true, y_pred, opts)
    end
  end

  @doc ~S"""
  Computes area under the curve (AUC) using the trapezoidal rule.

  This is a general function, given points on a curve.

  ## Examples

      iex> y = Nx.tensor([0, 0, 1, 1])
      iex> pred = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> distinct_value_indices = Scholar.Metrics.Classification.distinct_value_indices(pred)
      iex> {fpr, tpr, _thresholds} = Scholar.Metrics.Classification.roc_curve(y, pred, distinct_value_indices)
      iex> Scholar.Metrics.Classification.auc(fpr, tpr)
      #Nx.Tensor<
        f32
        0.75
      >
  """
  defn auc(x, y) do
    check_shape(x, y)
    dx = Nx.diff(x)

    # 0 means x is neither increasing nor decreasing -> error
    direction =
      cond do
        Nx.all(dx <= 0) -> -1
        Nx.all(dx >= 0) -> 1
        true -> Nx.tensor(:nan, type: to_float_type(y))
      end

    direction * Integrate.trapezoidal(y, x)
  end

  @doc ~S"""
  It's a helper function for `Scholar.Metrics.Classification.roc_curve` and `Scholar.Metrics.Classification.roc_auc_score` functions.
  You should call it and use as follows:

      distinct_value_indices = Scholar.Metrics.Classification.distinct_value_indices(scores)
      {fpr, tpr, thresholds} = Scholar.Metrics.Classification.roc_curve(y_true, scores, distinct_value_indices, weights)
  """
  def distinct_value_indices(y_score) do
    desc_score_indices = Nx.argsort(y_score, direction: :desc)
    y_score = Nx.take_along_axis(y_score, desc_score_indices)

    distinct_value_indices_mask = Nx.not_equal(y_score[[1..-1//1]], y_score[[0..-2//1]])

    Nx.iota({Nx.size(y_score) - 1})
    |> Nx.add(1)
    |> Nx.multiply(distinct_value_indices_mask)
    |> Nx.subtract(1)
    |> Nx.to_flat_list()
    |> Enum.filter(fn x -> x != -1 end)
    |> Nx.tensor()
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

  @doc ~S"""
  Compute precision-recall pairs for different probability thresholds.

  Note: this implementation is restricted to the binary classification task.

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> distinct_value_indices = Scholar.Metrics.Classification.distinct_value_indices(scores)
      iex> weights = Nx.tensor([1, 1, 2, 2])
      iex> {precision, recall, thresholds} = Scholar.Metrics.Classification.precision_recall_curve(y_true, scores, distinct_value_indices, weights)
      iex> precision
      #Nx.Tensor<
        f32[5]
        [0.6666666865348816, 0.800000011920929, 0.6666666865348816, 1.0, 1.0]
      >
      iex> recall
      #Nx.Tensor<
        f32[5]
        [1.0, 1.0, 0.5, 0.5, 0.0]
      >
      iex> thresholds
      #Nx.Tensor<
        f32[4]
        [0.10000000149011612, 0.3499999940395355, 0.4000000059604645, 0.800000011920929]
      >
  """
  defn precision_recall_curve(
         y_true,
         probabilities_predicted,
         distinct_value_indices,
         weights \\ 1.0
       ) do
    num_samples = Nx.axis_size(y_true, 0)
    weights = validate_weights(weights, num_samples, type: to_float_type(y_true))

    {fps, tps, thresholds} =
      binary_clf_curve(y_true, probabilities_predicted, distinct_value_indices, weights)

    precision_denominator = Nx.select(tps + fps == 0, 1, tps + fps)
    precision = Nx.select(tps + fps == 0, 1, tps / precision_denominator)

    recall =
      if tps[[-1]] == 0.0,
        do: Nx.broadcast(Nx.tensor(1.0, type: Nx.type(tps)), tps),
        else: tps / tps[[-1]]

    {Nx.concatenate([Nx.reverse(precision), Nx.tensor([1])], axis: 0),
     Nx.concatenate([Nx.reverse(recall), Nx.tensor([0])], axis: 0), Nx.reverse(thresholds)}
  end

  @doc ~S"""
  Compute average precision (AP) from prediction scores.

  AP summarizes a precision-recall curve as the weighted mean of precisions achieved at
  each threshold, with the increase in recall from the previous threshold used as the weight:
  $$ AP = sum_n (R_n - R_{n-1}) P_n $$

  where $ P_n $ and $ R_n $ are the precision and recall at the nth threshold.

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> distinct_value_indices = Scholar.Metrics.Classification.distinct_value_indices(scores)
      iex> weights = Nx.tensor([1, 1, 2, 2])
      iex> ap = Scholar.Metrics.Classification.average_precision_score(y_true, scores, distinct_value_indices, weights)
      iex> ap
      #Nx.Tensor<
        f32
        0.8999999761581421
      >
  """
  defn average_precision_score(
         y_true,
         probabilities_predicted,
         distinct_value_indices,
         weights \\ 1.0
       ) do
    num_samples = Nx.axis_size(y_true, 0)
    weights = validate_weights(weights, num_samples, type: to_float_type(y_true))

    {precision, recall, _thresholds} =
      precision_recall_curve(y_true, probabilities_predicted, distinct_value_indices, weights)

    -Nx.sum(Nx.diff(recall) * precision[0..-2//1])
  end

  # TODO implement :drop_intermediate option when dynamic shapes will be available
  @doc ~S"""
  Compute Receiver operating characteristic (ROC).

  Note: this implementation is restricted to the binary classification task.

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> distinct_value_indices = Scholar.Metrics.Classification.distinct_value_indices(scores)
      iex> weights = Nx.tensor([1, 1, 2, 2])
      iex> {fpr, tpr, thresholds} = Scholar.Metrics.Classification.roc_curve(y_true, scores, distinct_value_indices, weights)
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
  defn roc_curve(y_true, y_score, distinct_value_indices, weights \\ 1.0) do
    num_samples = Nx.axis_size(y_true, 0)
    weights = validate_weights(weights, num_samples, type: to_float_type(y_true))

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
      iex> distinct_value_indices = Scholar.Metrics.Classification.distinct_value_indices(scores)
      iex> weights = Nx.tensor([1, 1, 2, 2])
      iex> Scholar.Metrics.Classification.roc_auc_score(y_true, scores, distinct_value_indices, weights)
      #Nx.Tensor<
        f32
        0.75
      >
  """
  defn roc_auc_score(y_true, y_score, distinct_value_indices, weights \\ 1.0) do
    num_samples = Nx.axis_size(y_true, 0)
    weights = validate_weights(weights, num_samples, type: to_float_type(y_true))
    {fpr, tpr, _} = roc_curve(y_true, y_score, distinct_value_indices, weights)
    auc(fpr, tpr)
  end

  @doc """
  Compute the Brier score loss.

  The smaller the Brier score loss, the better, hence the naming with "loss".
  The Brier score measures the mean squared difference between the predicted
  probability and the actual outcome. The Brier score always
  takes on a value between zero and one, since this is the largest
  possible difference between a predicted probability (which must be
  between zero and one) and the actual outcome (which can take on values
  of only 0 and 1). It can be decomposed as the sum of refinement loss and
  calibration loss. If predicted probabilities are not in the interval
  [0, 1], they will be clipped.

  The Brier score is appropriate only for binary outcomes.

  ## Options

  #{NimbleOptions.docs(@brier_score_loss_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 1, 0])
      iex> y_prob = Nx.tensor([0.1, 0.9, 0.8, 0.3])
      iex> Scholar.Metrics.Classification.brier_score_loss(y_true, y_prob)
      #Nx.Tensor<
        f32
        0.03750000149011612
      >
  """
  deftransform brier_score_loss(y_true, y_prob, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @brier_score_loss_schema)
    {pos_label, opts} = Keyword.pop!(opts, :pos_label)
    brier_score_loss_n(y_true, y_prob, pos_label, opts)
  end

  defnp brier_score_loss_n(y_true, y_prob, pos_label, opts) do
    y_prob = Nx.clip(y_prob, 0.0, 1.0)
    size = Nx.axis_size(y_true, 0)
    weights = validate_weights(opts[:sample_weights], size, type: to_float_type(y_true))
    y_true = y_true == pos_label
    Nx.weighted_mean((y_true - y_prob) ** 2, weights)
  end

  @doc """
  Compute Cohen's kappa: a statistic that measures inter-annotator agreement.

  ## Options

  #{NimbleOptions.docs(@cohen_kappa_schema)}

  ## Examples

      iex> y1 = Nx.tensor([0, 1, 1, 0, 1, 2])
      iex> y2 = Nx.tensor([0, 2, 1, 0, 0, 1])
      iex> Scholar.Metrics.Classification.cohen_kappa_score(y1, y2, num_classes: 3)
      #Nx.Tensor<
        f32
        0.21739131212234497
      >

      iex> y1 = Nx.tensor([0, 1, 1, 0, 1, 2])
      iex> y2 = Nx.tensor([0, 2, 1, 0, 0, 1])
      iex> Scholar.Metrics.Classification.cohen_kappa_score(y1, y2, num_classes: 3, weighting_type: :linear)
      #Nx.Tensor<
        f32
        0.3571428060531616
      >
  """
  deftransform cohen_kappa_score(y1, y2, opts \\ []) do
    cohen_kappa_score_n(y1, y2, NimbleOptions.validate!(opts, @cohen_kappa_schema))
  end

  defnp cohen_kappa_score_n(y1, y2, opts) do
    num_classes = opts[:num_classes]
    cm = confusion_matrix(y1, y2, sample_weights: opts[:sample_weights], num_classes: num_classes)
    sum0 = Nx.sum(cm, axes: [0])
    sum1 = Nx.sum(cm, axes: [1])
    expected = Nx.outer(sum0, sum1) / Nx.sum(sum0)

    weights_matrix =
      case opts[:weighting_type] do
        nil ->
          wm = Nx.broadcast(1, cm)
          wm - Nx.eye(Nx.shape(wm))

        :linear ->
          wm = Nx.tile(Nx.iota({num_classes}), [num_classes, 1])
          Nx.abs(wm - Nx.transpose(wm))

        :quadratic ->
          wm = Nx.tile(Nx.iota({num_classes}), [num_classes, 1])
          (wm - Nx.transpose(wm)) ** 2
      end

    1 - Nx.sum(weights_matrix * cm) / Nx.sum(weights_matrix * expected)
  end

  @doc """
  Computes the log loss, aka logistic loss or cross-entropy loss.

  The log-loss is a measure of how well a forecaster performs, with smaller
  values being better. For each sample, a forecaster outputs a probability for
  each class, from which the log loss is computed by averaging the negative log
  of the probability forecasted for the true class over a number of samples.

  `y_true` should contain `num_classes` unique values, and the sum of `y_prob`
  along axis 1 should be 1 to respect the law of total probability.

  ## Options

  #{NimbleOptions.docs(@log_loss_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> y_prob = Nx.tensor([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.01, 0.99]])
      iex> Scholar.Metrics.Classification.log_loss(y_true, y_prob, num_classes: 2)
      #Nx.Tensor<
        f32
        0.17380733788013458
      >
      iex> Scholar.Metrics.Classification.log_loss(y_true, y_prob, num_classes: 2, normalize: false)
      #Nx.Tensor<
        f32
        0.6952293515205383
      >
      iex> weights = Nx.tensor([0.7, 2.3, 1.3, 0.34])
      iex(361)> Scholar.Metrics.Classification.log_loss(y_true, y_prob, num_classes: 2, sample_weights: weights)
      #Nx.Tensor<
        f32
        0.22717177867889404
      >
  """
  deftransform log_loss(y_true, y_prob, opts \\ []) do
    log_loss_n(
      y_true,
      y_prob,
      NimbleOptions.validate!(opts, @log_loss_schema)
    )
  end

  defnp log_loss_n(y_true, y_prob, opts) do
    assert_rank!(y_true, 1)
    assert_rank!(y_prob, 2)

    if Nx.axis_size(y_true, 0) != Nx.axis_size(y_prob, 0) do
      raise ArgumentError, "y_true and y_prob must have the same size along axis 0"
    end

    num_classes = opts[:num_classes]

    if Nx.axis_size(y_prob, 1) != num_classes do
      raise ArgumentError, "y_prob must have a size of num_classes along axis 1"
    end

    weights =
      validate_weights(
        opts[:sample_weights],
        Nx.axis_size(y_true, 0),
        type: to_float_type(y_prob)
      )

    y_true_onehot =
      ordinal_encode(y_true, num_classes: num_classes)
      |> one_hot_encode(num_classes: num_classes)

    y_prob = Nx.clip(y_prob, 0, 1)

    sample_loss =
      Nx.multiply(y_true_onehot, y_prob)
      |> Nx.sum(axes: [-1])
      |> Nx.log()
      |> Nx.negate()

    if opts[:normalize] do
      Nx.weighted_mean(sample_loss, weights)
    else
      Nx.multiply(sample_loss, weights)
      |> Nx.sum()
    end
  end

  @doc """
  Top-k Accuracy classification score.

  This metric computes the number of times where the correct label is
  among the top k labels predicted (ranked by predicted scores).

  For binary task assumed that y_score have values from 0 to 1.

  ## Options

  #{NimbleOptions.docs(@top_k_accuracy_score_schema)}

  ## Examples

      iex> y_true = Nx.tensor([0, 1, 2, 2, 0])
      iex> y_score = Nx.tensor([[0.5, 0.2, 0.1], [0.3, 0.4, 0.5], [0.4, 0.3, 0.2], [0.1, 0.3, 0.6], [0.9, 0.1, 0.0]])
      iex> Scholar.Metrics.Classification.top_k_accuracy_score(y_true, y_score, k: 2, num_classes: 3)
      #Nx.Tensor<
        f32
        0.800000011920929
      >

      iex> y_true = Nx.tensor([0, 1, 2, 2, 0])
      iex> y_score = Nx.tensor([[0.5, 0.2, 0.1], [0.3, 0.4, 0.5], [0.4, 0.3, 0.2], [0.1, 0.3, 0.6], [0.9, 0.1, 0.0]])
      iex> Scholar.Metrics.Classification.top_k_accuracy_score(y_true, y_score, k: 2, num_classes: 3, normalize: false)
      #Nx.Tensor<
        u64
        4
      >

      iex> y_true = Nx.tensor([0, 1, 0, 1, 0])
      iex> y_score = Nx.tensor([0.55, 0.3, 0.1, -0.2, 0.99])
      iex> Scholar.Metrics.Classification.top_k_accuracy_score(y_true, y_score, k: 1, num_classes: 2)
      #Nx.Tensor<
        f32
        0.20000000298023224
      >
  """
  deftransform top_k_accuracy_score(y_true, y_prob, opts \\ []) do
    top_k_accuracy_score_n(
      y_true,
      y_prob,
      NimbleOptions.validate!(opts, @top_k_accuracy_score_schema)
    )
  end

  defnp top_k_accuracy_score_n(y_true, y_score, opts) do
    k = opts[:k]
    num_classes = opts[:num_classes]

    hits =
      case num_classes do
        1 ->
          raise ArgumentError, "num_classes must be greater than 1"

        2 ->
          if k == 1 do
            rank_1d(Nx.rank(y_score))
            threshold = 0.5
            y_pred = y_score > threshold
            y_pred == y_true
          else
            Nx.broadcast(Nx.u8(1), y_true)
          end

        _ ->
          check_num_classes(num_classes, Nx.axis_size(y_score, 1))
          sorted_pred = Nx.argsort(y_score, axis: 1, direction: :desc)

          Nx.any(Nx.new_axis(y_true, 0) == Nx.transpose(sorted_pred[[.., 0..(k - 1)//1]]),
            axes: [0]
          )
      end

    case opts[:normalize] do
      true -> Nx.mean(hits)
      false -> Nx.sum(hits)
    end
  end

  deftransformp check_num_classes(num_classes, axis_size) do
    if num_classes != axis_size do
      raise ArgumentError,
            "num_classes must be equal to the second axis size, got #{num_classes} != #{axis_size}"
    end
  end

  deftransformp rank_1d(rank) do
    if rank != 1 do
      raise ArgumentError, "For binary task rank of y_score must be 1, got #{rank}"
    end
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

  @doc """
  Matthews Correlation Coefficient (MCC) provides a measure of the quality of binary classifications.

  It returns a value between -1 and 1 where 1 represents a perfect prediction, 0 represents no better
  than random prediction, and -1 indicates total disagreement between prediction and observation.
  """
  defn mcc(y_true, y_pred) do
    true_positives = binary_true_positives(y_true, y_pred)
    true_negatives = binary_true_negatives(y_true, y_pred)
    false_positives = binary_false_positives(y_true, y_pred)
    false_negatives = binary_false_negatives(y_true, y_pred)

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
           true_negatives == zero_tensor
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
end
