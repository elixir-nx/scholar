defmodule Scholar.DiscriminantAnalysis.Linear do
  @moduledoc ~S"""
  Linear Discriminant Analysis (LDA) for classification.

  LDA fits a Gaussian density to each class, assuming that all classes share the
  same covariance matrix. The shared covariance leads to a linear decision
  boundary between every pair of classes. For a sample $x$ the score of class $k$
  is

  $$ \delta\_{k}(x) = x^{T} \Sigma^{-1} \mu\_{k} - \frac{1}{2} \mu\_{k}^{T} \Sigma^{-1} \mu\_{k} + \log \pi\_{k} $$

  where $\mu\_{k}$ is the class mean, $\pi\_{k}$ the class prior and $\Sigma$ the
  pooled within-class covariance. The predicted class is the one with the
  highest score.

  The pooled covariance is assumed to be invertible, which holds when the number
  of samples is larger than the number of features and no feature is a linear
  combination of the others.

  Reference:

  * [1] - [The Elements of Statistical Learning, Hastie, Tibshirani & Friedman, Section 4.3](https://hastie.su.domains/ElemStatLearn/)
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:coefficients, :intercept, :means, :priors, :classes]}
  defstruct [:coefficients, :intercept, :means, :priors, :classes]

  opts_schema = [
    num_classes: [
      type: :pos_integer,
      required: true,
      doc:
        "Number of different classes used in training. Labels must be integers in `[0, num_classes)`."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Fits a Linear Discriminant Analysis model for sample inputs `x` and target
  labels `y`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:coefficients` - Weight vector of each class, shape `{num_classes, num_features}`.

    * `:intercept` - Intercept term of each class, shape `{num_classes}`.

    * `:means` - Class-wise mean of the samples, shape `{num_classes, num_features}`.

    * `:priors` - Class prior probabilities, shape `{num_classes}`.

    * `:classes` - Labels seen during fit, shape `{num_classes}`.

  ## Examples

      iex> x = Nx.tensor([[-2.0, -1.0], [-1.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1, 1])
      iex> model = Scholar.DiscriminantAnalysis.Linear.fit(x, y, num_classes: 2)
      iex> Scholar.DiscriminantAnalysis.Linear.predict(model, Nx.tensor([[-1.5, -1.5], [1.5, 1.5]]))
      #Nx.Tensor<
        s32[2]
        [0, 1]
      >
  """
  deftransform fit(x, y, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {num_samples, num_features}, " <>
              "got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    if Nx.rank(y) != 1 do
      raise ArgumentError,
            "expected y to have shape {num_samples}, " <>
              "got tensor with shape: #{inspect(Nx.shape(y))}"
    end

    if Nx.axis_size(x, 0) != Nx.axis_size(y, 0) do
      raise ArgumentError,
            "expected x and y to have the same number of samples, " <>
              "got #{Nx.axis_size(x, 0)} and #{Nx.axis_size(y, 0)}"
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)
    fit_n(x, y, opts)
  end

  defnp fit_n(x, y, opts) do
    x = to_float(x)
    num_classes = opts[:num_classes]
    {num_samples, _num_features} = Nx.shape(x)
    classes = Nx.iota({num_classes})

    one_hot = (Nx.new_axis(y, 1) == Nx.new_axis(classes, 0)) |> Nx.as_type(Nx.type(x))
    class_count = Nx.sum(one_hot, axes: [0])
    priors = class_count / num_samples
    means = Nx.dot(one_hot, [0], x, [0]) / Nx.new_axis(class_count, 1)
    xbar = Nx.dot(priors, means)

    # Pooled within-class covariance: center each sample by its class mean.
    centered = x - Nx.take(means, y, axis: 0)
    covariance = Nx.dot(centered, [0], centered, [0]) / (num_samples - num_classes)

    centered_means = means - xbar
    coefficients = Nx.LinAlg.solve(covariance, Nx.transpose(centered_means)) |> Nx.transpose()

    intercept =
      -0.5 * Nx.sum(coefficients * centered_means, axes: [1]) + Nx.log(priors) -
        Nx.dot(coefficients, xbar)

    %__MODULE__{
      coefficients: coefficients,
      intercept: intercept,
      means: means,
      priors: priors,
      classes: classes
    }
  end

  @doc """
  Computes the linear discriminant scores of each class for samples `x`.

  ## Examples

      iex> x = Nx.tensor([[-2.0, -1.0], [-1.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1, 1])
      iex> model = Scholar.DiscriminantAnalysis.Linear.fit(x, y, num_classes: 2)
      iex> scores = Scholar.DiscriminantAnalysis.Linear.decision_function(model, Nx.tensor([[1.5, 1.5]]))
      iex> Nx.shape(scores)
      {1, 2}
  """
  defn decision_function(%__MODULE__{coefficients: coefficients, intercept: intercept}, x) do
    x = to_float(x)
    Nx.dot(x, [1], coefficients, [1]) + intercept
  end

  @doc """
  Predicts the class of each sample in `x`.

  ## Examples

      iex> x = Nx.tensor([[-2.0, -1.0], [-1.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1, 1])
      iex> model = Scholar.DiscriminantAnalysis.Linear.fit(x, y, num_classes: 2)
      iex> Scholar.DiscriminantAnalysis.Linear.predict(model, Nx.tensor([[-1.5, -1.5], [1.5, 1.5]]))
      #Nx.Tensor<
        s32[2]
        [0, 1]
      >
  """
  defn predict(%__MODULE__{classes: classes} = model, x) do
    scores = decision_function(model, x)
    Nx.take(classes, Nx.argmax(scores, axis: 1))
  end

  @doc """
  Estimates class probabilities for samples `x` by applying a softmax to the
  discriminant scores.

  ## Examples

      iex> x = Nx.tensor([[-2.0, -1.0], [-1.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1, 1])
      iex> model = Scholar.DiscriminantAnalysis.Linear.fit(x, y, num_classes: 2)
      iex> probs = Scholar.DiscriminantAnalysis.Linear.predict_probability(model, Nx.tensor([[1.5, 1.5]]))
      iex> Nx.sum(probs) |> Nx.round()
      #Nx.Tensor<
        f32
        1.0
      >
  """
  defn predict_probability(%__MODULE__{} = model, x) do
    scores = decision_function(model, x)
    scores = scores - Nx.reduce_max(scores, axes: [1], keep_axes: true)
    exp = Nx.exp(scores)
    exp / Nx.sum(exp, axes: [1], keep_axes: true)
  end
end
