defmodule Scholar.DiscriminantAnalysis.Quadratic do
  @moduledoc ~S"""
  Quadratic Discriminant Analysis (QDA) for classification.

  QDA fits a Gaussian density to each class, letting every class have its own
  covariance matrix. Unlike `Scholar.DiscriminantAnalysis.Linear`, which pools a
  single covariance across classes, the per-class covariances leave a quadratic
  term in the score, so the decision boundary between two classes is a quadric
  rather than a hyperplane. For a sample $x$ the score of class $k$ is

  $$ \delta\_{k}(x) = -\frac{1}{2} \log |\Sigma\_{k}| - \frac{1}{2} (x - \mu\_{k})^{T} \Sigma\_{k}^{-1} (x - \mu\_{k}) + \log \pi\_{k} $$

  where $\mu\_{k}$ is the class mean, $\Sigma\_{k}$ the class covariance and
  $\pi\_{k}$ the class prior. The predicted class is the one with the highest
  score.

  Each covariance is stored through its eigendecomposition $\Sigma\_{k} = R\_{k}
  \Lambda\_{k} R\_{k}^{T}$, which turns both the log-determinant and the
  quadratic form into elementwise work over the eigenvalues.

  Every class covariance is assumed to be invertible, which requires at least as
  many samples as features in each class and no feature being a linear
  combination of the others. Use `:reg_param` to shrink the eigenvalues towards
  one when that does not hold.

  Reference:

  * [1] - [The Elements of Statistical Learning, Hastie, Tibshirani & Friedman, Section 4.3](https://hastie.su.domains/ElemStatLearn/)
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:means, :priors, :scalings, :rotations]}
  defstruct [:means, :priors, :scalings, :rotations]

  # `Nx.LinAlg.eigh/2` defaults to `eps: 1.0e-4`, which leaves the eigenvalues
  # about three digits short. QDA divides by them and takes their log, so the
  # error is amplified rather than absorbed. Measured against LAPACK on the same
  # f64 covariance, 1.0e-8 lands the eigenvalues within ~1.0e-7; tightening
  # further makes it worse, since the iteration then runs into `:max_iter`.
  @eigh_eps 1.0e-8

  opts_schema = [
    num_classes: [
      type: :pos_integer,
      required: true,
      doc:
        "Number of different classes used in training. Labels must be integers in `[0, num_classes)`."
    ],
    reg_param: [
      type: {:or, [:float, :integer]},
      default: 0.0,
      doc: """
      Regularizes the per-class covariance by shrinking its eigenvalues towards one,
      as `(1 - reg_param) * eigenvalues + reg_param`. Use a value in `[0, 1]` when a
      class has fewer samples than features, or when features are collinear, which
      would otherwise leave the covariance singular.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Fits a Quadratic Discriminant Analysis model for sample inputs `x` and target
  labels `y`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:means` - Class-wise mean of the samples, shape `{num_classes, num_features}`.

    * `:priors` - Class prior probabilities, shape `{num_classes}`.

    * `:scalings` - Eigenvalues of each class covariance, shape `{num_classes, num_features}`.

    * `:rotations` - Eigenvectors of each class covariance, shape `{num_classes, num_features, num_features}`.

  ## Examples

      iex> x = Nx.tensor([[-2.0, -1.0], [-1.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1, 1])
      iex> model = Scholar.DiscriminantAnalysis.Quadratic.fit(x, y, num_classes: 2)
      iex> Scholar.DiscriminantAnalysis.Quadratic.predict(model, Nx.tensor([[-1.5, -1.5], [1.5, 1.5]]))
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
    reg_param = opts[:reg_param]
    {num_samples, num_features} = Nx.shape(x)

    one_hot =
      (Nx.new_axis(y, 1) == Nx.new_axis(Nx.iota({num_classes}), 0)) |> Nx.as_type(Nx.type(x))

    class_count = Nx.sum(one_hot, axes: [0])
    priors = class_count / num_samples
    means = Nx.dot(one_hot, [0], x, [0]) / Nx.new_axis(class_count, 1)

    # One covariance per class. Looping keeps the working set at O(num_samples *
    # num_features); masking every class at once would need a
    # {num_classes, num_samples, num_features} intermediate instead.
    covariances =
      Nx.broadcast(Nx.tensor(0, type: Nx.type(x)), {num_classes, num_features, num_features})

    {covariances, _} =
      while {covariances, {x, one_hot, means, k = 0}}, k < num_classes do
        mask = Nx.new_axis(one_hot[[.., k]], 1)
        centered = (x - means[k]) * mask
        covariance = Nx.dot(centered, [0], centered, [0]) / (Nx.sum(mask) - 1)

        covariances = Nx.put_slice(covariances, [k, 0, 0], Nx.new_axis(covariance, 0))
        {covariances, {x, one_hot, means, k + 1}}
      end

    {scalings, rotations} = Nx.LinAlg.eigh(covariances, eps: @eigh_eps)
    scalings = (1 - reg_param) * scalings + reg_param

    %__MODULE__{
      means: means,
      priors: priors,
      scalings: scalings,
      rotations: rotations
    }
  end

  @doc """
  Computes the quadratic discriminant scores of each class for samples `x`.

  ## Examples

      iex> x = Nx.tensor([[-2.0, -1.0], [-1.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1, 1])
      iex> model = Scholar.DiscriminantAnalysis.Quadratic.fit(x, y, num_classes: 2)
      iex> scores = Scholar.DiscriminantAnalysis.Quadratic.decision_function(model, Nx.tensor([[1.5, 1.5]]))
      iex> Nx.shape(scores)
      {1, 2}
  """
  defn decision_function(
         %__MODULE__{means: means, priors: priors, scalings: scalings, rotations: rotations},
         x
       ) do
    x = to_float(x)

    # {num_classes, num_samples, num_features}
    centered = Nx.new_axis(x, 0) - Nx.new_axis(means, 1)
    rotated = Nx.dot(centered, [2], [0], rotations, [1], [0])

    mahalanobis = Nx.sum(rotated ** 2 / Nx.new_axis(scalings, 1), axes: [2])
    log_det = Nx.sum(Nx.log(scalings), axes: [1])

    scores = -0.5 * (mahalanobis + Nx.new_axis(log_det, 1)) + Nx.new_axis(Nx.log(priors), 1)
    Nx.transpose(scores)
  end

  @doc """
  Predicts the class of each sample in `x`.

  ## Examples

      iex> x = Nx.tensor([[-2.0, -1.0], [-1.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1, 1])
      iex> model = Scholar.DiscriminantAnalysis.Quadratic.fit(x, y, num_classes: 2)
      iex> Scholar.DiscriminantAnalysis.Quadratic.predict(model, Nx.tensor([[-1.5, -1.5], [1.5, 1.5]]))
      #Nx.Tensor<
        s32[2]
        [0, 1]
      >
  """
  defn predict(%__MODULE__{} = model, x) do
    scores = decision_function(model, x)
    Nx.argmax(scores, axis: 1)
  end

  @doc """
  Estimates class probabilities for samples `x` by applying a softmax to the
  discriminant scores.

  ## Examples

      iex> x = Nx.tensor([[-2.0, -1.0], [-1.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1, 1])
      iex> model = Scholar.DiscriminantAnalysis.Quadratic.fit(x, y, num_classes: 2)
      iex> probs = Scholar.DiscriminantAnalysis.Quadratic.predict_probability(model, Nx.tensor([[1.5, 1.5]]))
      iex> Nx.sum(probs) |> Nx.round()
      #Nx.Tensor<
        f32
        1.0
      >
  """
  defn predict_probability(%__MODULE__{} = model, x) do
    scores = decision_function(model, x)
    scores = scores - stop_grad(Nx.reduce_max(scores, axes: [1], keep_axes: true))
    exp = Nx.exp(scores)
    exp / Nx.sum(exp, axes: [1], keep_axes: true)
  end
end
