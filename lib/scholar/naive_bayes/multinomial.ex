defmodule Scholar.NaiveBayes.Multinomial do
  @moduledoc """
  Naive Bayes classifier for multinomial models.

  Time complexity is $O(K * N * C)$ where $N$ is the number of samples and $K$ is the number of features,
  and $C$ is the number of classes.
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container,
           containers: [
             :feature_count,
             :class_count,
             :class_log_priors,
             :classes,
             :feature_log_probability
           ]}
  defstruct [:feature_count, :class_count, :class_log_priors, :classes, :feature_log_probability]

  opts_schema = [
    alpha: [
      type: {:or, [:float, {:list, :float}]},
      default: 1.0,
      doc: ~S"""
      Additive (Laplace/Lidstone) smoothing parameter
      (set alpha to 0.0 and force_alpha to true, for no smoothing).
      """
    ],
    force_alpha: [
      type: :boolean,
      default: true,
      doc: ~S"""
      If `false` and alpha is less than 1e-10, it will set alpha to
      1e-10. If `true`, alpha will remain unchanged. This may cause
      numerical errors if alpha is too close to 0.
      """
    ],
    fit_priors: [
      type: :boolean,
      default: true,
      doc: ~S"""
      Whether to learn class prior probabilities or not.
      If `false`, a uniform prior will be used.
      """
    ],
    priors: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: ~S"""
      Prior probabilities of the classes. If specified, the priors are not
      adjusted according to the data.
      """
    ],
    num_classes: [
      type: :pos_integer,
      required: true,
      doc: ~S"""
      Number of different classes used in training.
      """
    ],
    sample_weights: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: ~S"""
      List of `n_samples` elements.
      A list of 1.0 values is used if none is given.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  The multinomial Naive Bayes classifier is suitable for classification with
  discrete features (e.g., word counts for text classification)

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:feature_log_probability` - Empirical log probability of features
        given a class, ``P(x_i|y)``.

    * `:class_count` - Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    * `:class_log_priors` - Smoothed empirical log probability for each class.

    * `:classes` - class labels known to the classifier.

    * `:feature_count` - Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 3)
      %Scholar.NaiveBayes.Multinomial{
        feature_log_probability: Nx.tensor(
          [
            [-1.232143759727478, -1.0986123085021973, -0.9808292388916016],
            [-1.7917594909667969, -1.0986123085021973, -0.6931471824645996],
            [-1.241713285446167, -1.0986123085021973, -0.9734492301940918]
          ]
        ),
        class_count: Nx.tensor([1.0, 1.0, 2.0]),
        class_log_priors: Nx.tensor([-1.3862943649291992, -1.3862943649291992, -0.6931471824645996]),
        classes: Nx.tensor([0, 1, 2]),
        feature_count: Nx.tensor(
          [
            [6.0, 7.0, 8.0],
            [0.0, 1.0, 2.0],
            [12.0, 14.0, 16.0]
          ]
        )
      }
      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 3, sample_weights: [1, 6, 2, 3])
      %Scholar.NaiveBayes.Multinomial{
        feature_log_probability: Nx.tensor(
          [
            [-1.241713285446167, -1.0986123085021973, -0.9734492301940918],
            [-1.7917594909667969, -1.0986123085021973, -0.6931471824645996],
            [-1.2773041725158691, -1.0986123085021973, -0.9470624923706055]
          ]
        ),
        class_count: Nx.tensor([2.0, 1.0, 9.0]),
        class_log_priors: Nx.tensor([-1.7917594909667969, -2.4849066734313965, -0.28768205642700195]),
        classes: Nx.tensor([0, 1, 2]),
        feature_count: Nx.tensor(
          [
            [12.0, 14.0, 16.0],
            [0.0, 1.0, 2.0],
            [45.0, 54.0, 63.0]
          ]
        )
      }
  """
  deftransform fit(x, y, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    opts =
      [
        sample_weights_flag: opts[:sample_weights] != nil,
        priors_flag: opts[:priors] != nil
      ] ++ opts

    x_type = to_float_type(x)

    {sample_weights, opts} = Keyword.pop(opts, :sample_weights, Nx.tensor(1.0, type: x_type))
    sample_weights = Nx.tensor(sample_weights, type: x_type)

    {priors, opts} = Keyword.pop(opts, :priors, Nx.tensor(0.0, type: x_type))
    class_priors = Nx.tensor(priors)
    {alpha, opts} = Keyword.pop!(opts, :alpha)
    alpha = Nx.tensor(alpha, type: x_type)

    fit_n(x, y, sample_weights, class_priors, alpha, opts)
  end

  @doc """
  Perform classification on an array of test vectors `x` using `model`.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Multinomial.predict(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        s64[2]
        [2, 2]
      >
  """
  defn predict(%__MODULE__{classes: classes} = model, x) do
    check_input(model, x)
    jll = joint_log_likelihood(model, x)
    Nx.take(classes, Nx.argmax(jll, axis: 1))
  end

  @doc """
  Return log-probability estimates for the test vector `x` using `model`.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Multinomial.predict_log_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [-1.1167821884155273, -3.3237485885620117, -0.45153331756591797],
          [-1.141427993774414, -3.029214859008789, -0.4584178924560547]
        ]
      >
  """
  defn predict_log_probability(%__MODULE__{} = model, x) do
    check_input(model, x)
    jll = joint_log_likelihood(model, x)

    log_proba_x =
      jll
      |> Nx.exp()
      |> Nx.sum(axes: [1])
      |> Nx.log()
      |> Nx.reshape({:auto, 1})
      |> Nx.broadcast(jll)

    jll - log_proba_x
  end

  @doc """
  Return probability estimates for the test vector `x` using `model`.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Multinomial.predict_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [0.32733139395713806, 0.036017563194036484, 0.6366512179374695],
          [0.3193626403808594, 0.048353586345911026, 0.6322832107543945]
        ]
      >
  """
  defn predict_probability(%__MODULE__{} = model, x) do
    Nx.exp(predict_log_probability(model, x))
  end

  @doc """
  Return joint log probability estimates for the test vector `x` using `model`.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Multinomial.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Multinomial.predict_joint_log_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [-14.899698257446289, -17.106664657592773, -14.23444938659668],
          [-25.563968658447266, -27.45175552368164, -24.880958557128906]
        ]
      >
  """
  defn predict_joint_log_probability(%__MODULE__{} = model, x) do
    check_input(model, x)
    joint_log_likelihood(model, x)
  end

  defnp fit_n(x, y, sample_weights, class_priors, alpha, opts) do
    x_type = Nx.Type.merge(to_float_type(x), {:f, 32})
    input_rank = Nx.rank(x)
    targets_rank = Nx.rank(y)

    if input_rank != 2 do
      raise ArgumentError,
            "wrong input rank. Expected x to be rank 2 got: #{input_rank}"
    end

    if targets_rank != 1 do
      raise ArgumentError,
            "wrong target rank. Expected target to be rank 1 got: #{targets_rank}"
    end

    {num_samples, num_features} = Nx.shape(x)
    {num_targets} = Nx.shape(y)

    if num_samples != num_targets do
      raise ArgumentError,
            "wrong input shape. Expected x to have the same first dimension as y, got: #{num_samples} for x and #{num_targets} for y"
    end

    num_classes = opts[:num_classes]

    class_priors =
      case Nx.shape(class_priors) do
        {} ->
          Nx.broadcast(class_priors, {num_classes})

        {^num_classes} ->
          class_priors

        _ ->
          raise ArgumentError,
                "number of priors must match number of classes. Number of priors: #{Nx.size(class_priors)} does not match number of classes: #{num_classes}"
      end

    sample_weights =
      case Nx.shape(sample_weights) do
        {} ->
          Nx.broadcast(sample_weights, {num_samples})

        {^num_samples} ->
          sample_weights

        _ ->
          raise ArgumentError,
                "number of weights must match number of samples. Number of weights: #{Nx.size(sample_weights)} does not match number of samples: #{num_samples}"
      end

    classes_encoded = Nx.iota({num_classes})

    classes =
      y
      |> Scholar.Preprocessing.ordinal_encode(num_classes: num_classes)
      |> Scholar.Preprocessing.one_hot_encode(num_classes: num_classes)

    {_, classes_features} = Nx.shape(classes)

    classes =
      cond do
        classes_features == 1 and num_classes == 2 ->
          Nx.concatenate([1 - classes, classes], axis: 1)

        classes_features == 1 and num_classes != 2 ->
          Nx.broadcast(1.0, Nx.shape(classes))

        true ->
          classes
      end

    classes =
      if opts[:sample_weights_flag],
        do: classes * Nx.reshape(sample_weights, {:auto, 1}),
        else: classes

    {_, n_classes} = Nx.shape(classes)
    class_count = Nx.broadcast(Nx.tensor(0.0, type: x_type), {n_classes})
    feature_count = Nx.broadcast(Nx.tensor(0.0, type: x_type), {n_classes, num_features})
    feature_count = feature_count + Nx.dot(classes, [0], x, [0])
    class_count = class_count + Nx.sum(classes, axes: [0])
    alpha = check_alpha(alpha, opts[:force_alpha], num_features)
    smoothed_feature_count = feature_count + alpha
    smoothed_cumulative_count = Nx.sum(smoothed_feature_count, axes: [1])

    feature_log_probability =
      Nx.log(smoothed_feature_count) - Nx.log(Nx.reshape(smoothed_cumulative_count, {:auto, 1}))

    {class_priors_length} = Nx.shape(class_priors)

    if num_classes != class_priors_length do
      raise ArgumentError, "Number of priors must match number of classes."
    end

    class_log_priors =
      cond do
        opts[:priors_flag] ->
          Nx.log(class_priors)

        opts[:fit_priors] ->
          Nx.log(class_count) - Nx.log(Nx.sum(class_count))

        true ->
          Nx.broadcast(-Nx.log(num_classes), {num_classes})
      end

    %__MODULE__{
      classes: classes_encoded,
      class_count: class_count,
      feature_log_probability: feature_log_probability,
      feature_count: feature_count,
      class_log_priors: class_log_priors
    }
  end

  defnp joint_log_likelihood(
          %__MODULE__{
            feature_log_probability: feature_log_probability,
            class_log_priors: class_log_priors
          },
          x
        ) do
    Nx.dot(x, [1], feature_log_probability, [1]) + class_log_priors
  end

  defnp check_alpha(alpha, force_alpha, num_features) do
    type = Nx.Type.merge(Nx.type(alpha), {:f, 32})
    alpha_lower_bound = Nx.tensor(1.0e-10, type: type)

    case Nx.shape(alpha) do
      {} -> nil
      {^num_features} -> nil
      _ -> raise ArgumentError, "when alpha is a list it should contain num_features values"
    end

    if force_alpha, do: alpha, else: Nx.max(alpha, alpha_lower_bound)
  end

  defnp check_input(%__MODULE__{feature_count: feature_count}, x) do
    num_features = Nx.axis_size(feature_count, 1)
    x_num_features = Nx.axis_size(x, 1)

    if num_features != x_num_features do
      raise ArgumentError,
            "wrong input shape. Expected x to have the same second dimension as the data for fitting process, got: #{x_num_features} for x and #{num_features} for training data"
    end
  end
end
