defmodule Scholar.NaiveBayes.Bernoulli do
  @moduledoc """
  Naive Bayes classifier for multivariate Bernoulli models.
  Like MultinomialNB, this classifier is suitable for discrete data. The
  difference is that while MultinomialNB works with occurrence counts,
  BernoulliNB is designed for binary/boolean features.
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container,
           containers: [
             :feature_count,
             :class_count,
             :class_log_priors,
             :feature_log_probability
           ]}
  defstruct [:feature_count, :class_count, :class_log_priors, :feature_log_probability]

  opts_schema = [
    num_classes: [
      type: :pos_integer,
      required: true,
      doc: ~S"""
      Number of different classes used in training.
      """
    ],
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
    binarize: [
      type: {:or, [:float, {:in, [nil]}]},
      default: 0.0,
      doc: ~S"""
      Threshold for binarizing (mapping to booleans) of sample features.
      If nil, input is presumed to already consist of binary vectors.
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
    class_priors: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: ~S"""
      Prior probabilities of the classes. If specified, the priors are not
      adjusted according to the data.
      """
    ],
    sample_weights: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: ~S"""
      List of `num_samples` elements.
      A list of 1.0 values is used if none is given.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Fits a naive Bayes model. The function assumes that the targets `y` are integers
  between 0 and `num_classes` - 1 (inclusive). Otherwise, those samples will not
  contribute to `class_count`.
  ## Options
  #{NimbleOptions.docs(@opts_schema)}
  ## Return Values
  The function returns a struct with the following parameters:
    * `:class_count` - Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.
    * `:class_log_priors` - Smoothed empirical log probability for each class.
    * `:feature_count` - Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.
    * `:feature_log_probability` - Empirical log probability of features
        given a class, ``P(x_i|y)``.
  ## Examples
      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> Scholar.NaiveBayes.Bernoulli.fit(x, y, num_classes: 3, binarize: 1.0)
      %Scholar.NaiveBayes.Bernoulli{
            feature_count: Nx.tensor(
              [
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [2.0, 2.0, 2.0]
              ]
            ),
            class_count: Nx.tensor(
              [1.0, 1.0, 2.0]
            ),
            class_log_priors: Nx.tensor(
              [-1.3862943649291992, -1.3862943649291992, -0.6931471824645996]
            ),
            feature_log_probability: Nx.tensor(
              [
                [-0.40546512603759766, -0.40546512603759766, -0.40546512603759766],
                [-1.0986123085021973, -1.0986123085021973, -0.40546512603759766],
                [-0.28768205642700195, -0.28768205642700195, -0.28768205642700195]
              ]
            )
          }
      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> Scholar.NaiveBayes.Bernoulli.fit(x, y, num_classes: 3, force_alpha: false, alpha: 0.0)
      %Scholar.NaiveBayes.Bernoulli{
            feature_count: Nx.tensor(
              [
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [2.0, 2.0, 2.0]
              ]
            ),
            class_count: Nx.tensor(
              [1.0, 1.0, 2.0]
            ),
            class_log_priors: Nx.tensor(
              [-1.3862943649291992, -1.3862943649291992, -0.6931471824645996]
            ),
            feature_log_probability: Nx.tensor(
              [
                [0.0, 0.0, 0.0],
                [-23.025850296020508, 0.0, 0.0],
                [0.0, 0.0, 0.0]
              ]
            )
          }
  """
  deftransform fit(x, y, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected x to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}\
            """
    end

    if Nx.rank(y) != 1 do
      raise ArgumentError,
            """
            expected y to have shape {num_samples}, \
            got tensor with shape: #{inspect(Nx.shape(y))}\
            """
    end

    {num_samples, num_features} = Nx.shape(x)

    if num_samples != Nx.axis_size(y, 0) do
      raise ArgumentError,
            """
            expected first dimension of x and y to be of same size, \
            got: #{num_samples} and #{Nx.axis_size(y, 0)}\
            """
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)
    type = to_float_type(x)

    x_binarize =
      if opts[:binarize] != nil,
        do: Scholar.Preprocessing.Binarizer.fit_transform(x, threshold: opts[:binarize]),
        else: x

    {alpha, opts} = Keyword.pop!(opts, :alpha)
    alpha = Nx.tensor(alpha, type: type)

    if Nx.shape(alpha) not in [{}, {num_features}] do
      raise ArgumentError,
            """
            when alpha is list it should have length equal to num_features = #{num_features}, \
            got: #{Nx.size(alpha)}\
            """
    end

    num_classes = opts[:num_classes]

    priors_flag = opts[:class_priors] != nil

    {class_priors, opts} = Keyword.pop(opts, :class_priors, :nan)
    class_priors = Nx.tensor(class_priors)

    if priors_flag and Nx.size(class_priors) != num_classes do
      raise ArgumentError,
            """
            expected class_priors to be list of length num_classes = #{num_classes}, \
            got: #{Nx.size(class_priors)}\
            """
    end

    sample_weights_flag = opts[:sample_weights] != nil

    {sample_weights, opts} = Keyword.pop(opts, :sample_weights, :nan)
    sample_weights = Nx.tensor(sample_weights, type: type)

    if sample_weights_flag and Nx.shape(sample_weights) != {num_samples} do
      raise ArgumentError,
            """
            expected sample_weights to be list of length num_samples = #{num_samples}, \
            got: #{Nx.size(sample_weights)}\
            """
    end

    opts =
      opts ++
        [
          type: type,
          priors_flag: priors_flag,
          sample_weights_flag: sample_weights_flag
        ]

    fit_n(x_binarize, y, class_priors, sample_weights, alpha, opts)
  end

  defnp fit_n(x, y, class_priors, sample_weights, alpha, opts) do
    type = opts[:type]
    num_samples = Nx.axis_size(x, 0)

    num_classes = opts[:num_classes]

    y_one_hot = Scholar.Preprocessing.OneHotEncoder.fit_transform(y, num_categories: num_classes)
    y_one_hot = Nx.select(y_one_hot, Nx.tensor(1, type: type), Nx.tensor(0, type: type))

    y_weighted =
      if opts[:sample_weights_flag],
        do: Nx.reshape(sample_weights, {num_samples, 1}) * y_one_hot,
        else: y_one_hot

    alpha_lower_bound = Nx.tensor(1.0e-10, type: type)

    alpha =
      if opts[:force_alpha], do: alpha, else: Nx.max(alpha, alpha_lower_bound)

    class_count = Nx.sum(y_weighted, axes: [0])
    feature_count = Nx.dot(y_weighted, [0], x, [0])

    smoothed_feature_count = feature_count + alpha
    smoothed_cumulative_count = class_count + alpha * 2

    feature_log_probability =
      Nx.log(smoothed_feature_count) -
        Nx.log(Nx.reshape(smoothed_cumulative_count, {num_classes, 1}))

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
      class_count: class_count,
      class_log_priors: class_log_priors,
      feature_count: feature_count,
      feature_log_probability: feature_log_probability
    }
  end

  @doc """
  Perform classification on an array of test vectors `x` using `model`.
  You need to add sorted classes from the training data as the second argument.
  ## Examples
      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Bernoulli.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Bernoulli.predict(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]), Nx.tensor([0, 1, 2]))
      #Nx.Tensor<
        s64[2]
        [2, 2]
      >
  """
  defn predict(%__MODULE__{} = model, x, classes) do
    check_dim(x, Nx.axis_size(model.feature_count, 1))

    if Nx.rank(classes) != 1 do
      raise ArgumentError,
            """
            expected classes to be a 1D tensor, \
            got tensor with shape: #{inspect(Nx.shape(classes))}\
            """
    end

    if Nx.axis_size(classes, 0) != Nx.axis_size(model.class_count, 0) do
      raise ArgumentError,
            """
            expected classes to have same size as the number of classes in the model, \
            got: #{Nx.axis_size(classes, 0)} for classes and #{Nx.axis_size(model.class_count, 0)} for model\
            """
    end

    jll = joint_log_likelihood(model, x)
    classes[Nx.argmax(jll, axis: 1)]
  end

  @doc """
  Return log-probability estimates for the test vector `x` using `model`.
  ## Examples
      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Bernoulli.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Bernoulli.predict_log_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [-4.704780578613281, -12.329399108886719, -0.009097099304199219],
          [-8.750494003295898, -19.147701263427734, -1.583099365234375e-4]
        ]
      >
  """
  defn predict_log_probability(%__MODULE__{} = model, x) do
    check_dim(x, Nx.axis_size(model.feature_count, 1))
    jll = joint_log_likelihood(model, x)

    log_proba_x =
      jll
      |> Nx.logsumexp(axes: [1])
      |> Nx.new_axis(1)
      |> Nx.broadcast(jll)

    jll - log_proba_x
  end

  @doc """
  Return probability estimates for the test vector `x` using `model`.
  ## Examples
      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Bernoulli.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Bernoulli.predict_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [0.00905190035700798, 4.4198750401847064e-6, 0.9909441471099854],
          [1.5838305989746004e-4, 4.833469624543341e-9, 0.9998416900634766]
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
      iex> model = Scholar.NaiveBayes.Bernoulli.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Bernoulli.predict_joint_log_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [3.6356334686279297, -3.988985061645508, 8.331316947937012],
          [10.56710433959961, 0.16989731788635254, 19.317440032958984]
        ]
      >
  """
  defn predict_joint_log_probability(%__MODULE__{} = model, x) do
    check_dim(x, Nx.axis_size(model.feature_count, 1))
    joint_log_likelihood(model, x)
  end

  defnp check_dim(x, dim) do
    num_features = Nx.axis_size(x, 1)

    if num_features != dim do
      raise ArgumentError,
            """
            expected x to have same second dimension as data used for fitting model, \
            got: #{num_features} for x and #{dim} for training data\
            """
    end
  end

  defnp joint_log_likelihood(
          %__MODULE__{
            feature_log_probability: feature_log_probability,
            class_log_priors: class_log_priors
          },
          x
        ) do
    neg_prob = Nx.log(1 - Nx.exp(feature_log_probability))
    jll = Nx.dot(x, [1], feature_log_probability - neg_prob, [1])
    jll + class_log_priors + Nx.sum(neg_prob, axes: [1])
  end
end
