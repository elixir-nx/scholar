defmodule Scholar.NaiveBayes.Categorical do
  @moduledoc """
  Naive Bayes classifier for categorical features.

  The categorical Naive Bayes classifier is suitable for classification with
  discrete features that are categorically distributed. The categories of
  each feature are drawn from a categorical distribution.
  """
  require Nx.Defn.Kernel
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
    ],
    min_categories: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: ~S"""
      List of minimum number of categories per feature.
      Determines the number of categories automatically from the training data if none is given.
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

    * `:feature_count` - A (num_features, num_classes, num_categories) tensor tracking the weighted count of each (feature, class, category) combination.
        Calculated by summing the weighted occurrences of feature values for each class-label during fitting.

    * `:feature_log_probability` - Empirical log probability of features
        given a class, ``P(x_i|y)``.

  ## Examples

      iex> x = Nx.tensor([[1, 2, 2], [1, 2, 1], [2, 2, 0]])
      iex> y = Nx.tensor([0, 1, 1])
      iex> Scholar.NaiveBayes.Categorical.fit(x, y, num_classes: 2)
      %Scholar.NaiveBayes.Categorical{
        feature_count: Nx.tensor(
          [
            [
              [0.0, 1.0, 0.0],
              [0.0, 1.0, 1.0]
            ],
            [
              [0.0, 0.0, 1.0],
              [0.0, 0.0, 2.0]
            ],
            [
              [0.0, 0.0, 1.0],
              [1.0, 1.0, 0.0]
            ]
          ]
        ),
        class_count: Nx.tensor([1.0, 2.0]),
        class_log_priors: Nx.tensor([-1.0986123085021973, -0.40546512603759766]),
        feature_log_probability: Nx.tensor(
          [
            [
              [-1.3862943649291992, -0.6931471824645996, -1.3862943649291992],
              [-1.6094379425048828, -0.9162907600402832, -0.9162907600402832]
            ],
            [
              [-1.3862943649291992, -1.3862943649291992, -0.6931471824645996],
              [-1.6094379425048828, -1.6094379425048828, -0.5108256340026855]
            ],
            [
              [-1.3862943649291992, -1.3862943649291992, -0.6931471824645996],
              [-0.9162907600402832, -0.9162907600402832, -1.6094379425048828]
            ]
          ]
        )
      }

      iex> x = Nx.tensor([[1, 2, 2], [1, 2, 1], [2, 2, 0]])
      iex> y = Nx.tensor([0, 1, 1])
      iex> Scholar.NaiveBayes.Categorical.fit(x, y, num_classes: 2, force_alpha: false, alpha: 0.0)
      %Scholar.NaiveBayes.Categorical{
        feature_count: Nx.tensor(
          [
            [
              [0.0, 1.0, 0.0],
              [0.0, 1.0, 1.0]
            ],
            [
              [0.0, 0.0, 1.0],
              [0.0, 0.0, 2.0]
            ],
            [
              [0.0, 0.0, 1.0],
              [1.0, 1.0, 0.0]
            ]
          ]
        ),
        class_count: Nx.tensor(
          [1.0, 2.0]
        ),
        class_log_priors: Nx.tensor(
          [-1.0986123085021973, -0.40546512603759766]
        ),
        feature_log_probability: Nx.tensor(
          [
            [
              [-23.025850296020508, 0.0, -23.025850296020508],
              [-23.718997955322266, -0.6931471824645996, -0.6931471824645996]
            ],
            [
              [-23.025850296020508, -23.025850296020508, 0.0],
              [-23.718997955322266, -23.718997955322266, 0.0]
            ],
            [
              [-23.025850296020508, -23.025850296020508, 0.0],
              [-0.6931471824645996, -0.6931471824645996, -23.718997955322266]
            ]
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

    min_categories_flag = opts[:min_categories] != nil

    {min_categories, opts} = Keyword.pop(opts, :min_categories, :nan)
    min_categories = Nx.tensor(min_categories, type: type)

    if min_categories_flag and Nx.shape(min_categories) != {num_features} do
      raise ArgumentError,
            """
            expected min_categories to be list of length num_features = #{num_features}, \
            got: #{Nx.size(min_categories)}\
            """
    end

    num_categories =
      (opts[:min_categories] || x)
      |> Nx.reduce_max()
      |> Nx.add(1)
      |> Nx.to_number()

    opts =
      opts ++
        [
          num_categories: num_categories,
          type: type,
          priors_flag: priors_flag,
          sample_weights_flag: sample_weights_flag,
          min_categories_flag: min_categories_flag
        ]

    fit_n(x, y, class_priors, sample_weights, alpha, min_categories, opts)
  end

  defn fit_n(x, y, class_priors, sample_weights, alpha, min_categories, opts) do
    type = opts[:type]
    {num_samples, num_features} = Nx.shape(x)

    num_classes = opts[:num_classes]
    num_categories = opts[:num_categories]

    min_categories =
      if opts[:min_categories_flag], do: min_categories, else: Nx.reduce_max(x, axes: [0]) + 1

    y_one_hot =
      y
      |> Nx.new_axis(1)
      |> Nx.broadcast({num_samples, num_classes})
      |> Nx.equal(Nx.iota({num_samples, num_classes}, axis: 1))
      |> Nx.as_type(type)

    y_weighted =
      if opts[:sample_weights_flag],
        do: Nx.reshape(sample_weights, {num_samples, 1}) * y_one_hot,
        else: y_one_hot

    alpha_lower_bound = Nx.tensor(1.0e-10, type: type)

    alpha =
      if opts[:force_alpha], do: alpha, else: Nx.max(alpha, alpha_lower_bound)

    {class_count, feature_count} =
      count(x, y_weighted, num_features, num_classes, num_categories, num_samples)

    feature_log_probability =
      compute_feature_log_probability(feature_count, alpha, min_categories)

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
      iex> model = Scholar.NaiveBayes.Categorical.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Categorical.predict(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]), Nx.tensor([0, 1, 2]))
      #Nx.Tensor<
        s32[2]
        [0, 2]
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
      iex> model = Scholar.NaiveBayes.Categorical.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Categorical.predict_log_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [-0.8266787528991699, -1.5198254585266113, -1.0678410530090332],
          [-1.272965431213379, -1.272965431213379, -0.8209810256958008]
        ]
      >
  """

  defn predict_log_probability(%__MODULE__{} = model, x) do
    check_dim(x, Nx.axis_size(model.feature_count, 1))
    jll = joint_log_likelihood(model, x)

    log_proba_x =
      jll
      |> Nx.logsumexp(axes: [1])
      |> Nx.reshape({Nx.axis_size(jll, 0), 1})

    jll - log_proba_x
  end

  @doc """
  Return probability estimates for the test vector `x` using `model`.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Categorical.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Categorical.predict_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [0.43749991059303284, 0.21875005960464478, 0.34374985098838806],
          [0.28000006079673767, 0.28000006079673767, 0.4399997889995575]
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
      iex> model = Scholar.NaiveBayes.Categorical.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Categorical.predict_joint_log_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [-8.140898704528809, -8.83404541015625, -8.382061004638672],
          [-8.83404541015625, -8.83404541015625, -8.382061004638672]
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
    {_, jll} =
      while {{i = 0, feature_log_probability, x}, jll = Nx.broadcast(0.0, Nx.shape(x))},
            i < Nx.axis_size(x, 1) do
        indices = Nx.slice_along_axis(x, i, 1, axis: 1) |> Nx.squeeze(axes: [1])

        jll =
          Nx.slice_along_axis(feature_log_probability, i, 1, axis: 0)
          |> Nx.squeeze()
          |> Nx.take(indices, axis: 1)
          |> Nx.transpose()
          |> Nx.add(jll)

        {{i + 1, feature_log_probability, x}, jll}
      end

    jll + class_log_priors
  end

  defnp count(x, y_weighted, num_features, num_classes, num_categories, num_samples) do
    class_count = Nx.sum(y_weighted, axes: [0])

    feature_count = Nx.broadcast(0.0, {num_features, num_classes, num_categories})

    {_, feature_count} =
      while {{i = 0, x, y_weighted, num_features}, feature_count}, i < num_samples do
        {_, feature_count} =
          while {{j = 0, x, y_weighted, i, num_features}, feature_count}, j < num_features do
            category_value = x[i][j]
            class_label = Nx.argmax(y_weighted[i])

            index = Nx.stack([j, class_label, category_value])

            feature_count = Nx.indexed_add(feature_count, index, y_weighted[i][class_label])

            {{j + 1, x, y_weighted, i, num_features}, feature_count}
          end

        {{i + 1, x, y_weighted, num_features}, feature_count}
      end

    {class_count, feature_count}
  end

  defnp compute_feature_log_probability(feature_count, alpha, min_categories) do
    feature_log_probability = Nx.broadcast(0.0, Nx.shape(feature_count))

    {_, feature_log_probability} =
      while {{i = 0, feature_count, alpha, min_categories}, feature_log_probability},
            i < Nx.axis_size(feature_count, 0) do
        smoothed_class_count =
          Nx.sum(feature_count[i], axes: [1])
          |> Nx.add(alpha * min_categories[i])
          |> Nx.log()
          |> Nx.new_axis(1)

        smoothed_cat_count =
          feature_count[i]
          |> Nx.add(alpha)
          |> Nx.log()
          |> Nx.subtract(smoothed_class_count)

        smoothed_cat_count =
          Nx.iota({Nx.axis_size(feature_count[i], 1)})
          |> Nx.less(min_categories[i])
          |> Nx.broadcast(Nx.shape(feature_count[i]))
          |> Nx.select(smoothed_cat_count, feature_count[i])
          |> Nx.new_axis(0)

        feature_log_probability =
          Nx.put_slice(feature_log_probability, [i, 0, 0], smoothed_cat_count)

        {{i + 1, feature_count, alpha, min_categories}, feature_log_probability}
      end

    feature_log_probability
  end
end
