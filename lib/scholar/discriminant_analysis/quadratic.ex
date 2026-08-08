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
  combination of the others. When that does not hold, a covariance is singular
  and its smallest eigenvalues come out as zero, which the score then divides by.
  Use `:reg_param` to shrink the eigenvalues towards one in that case.

  Note that scikit-learn handles rank deficiency differently: it decomposes the
  centered class data and keeps only `min(class_size, num_features)` components,
  so it works in the subspace the class actually spans. This implementation keeps
  all `num_features` directions and relies on `:reg_param` instead, so results
  diverge from scikit-learn's for a class with fewer samples than features.

  Forming the covariance squares its condition number, so features on wildly
  different scales cost accuracy in the eigendecomposition well before the
  matrix is singular. A compiled backend absorbs this, but the pure-Nx
  eigendecomposition does not, and features spanning several orders of magnitude
  can lose the smallest eigenvalues there entirely. Standardizing the features
  beforehand, with `Scholar.Preprocessing.StandardScaler`, avoids it either way.

  Reference:

  * [1] - [The Elements of Statistical Learning, Hastie, Tibshirani & Friedman, Section 4.3](https://hastie.su.domains/ElemStatLearn/)
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:means, :priors, :scalings, :rotations]}
  defstruct [:means, :priors, :scalings, :rotations]

  # Only matters for the pure-Nx eigendecomposition, which compilers like EXLA
  # replace with a native routine. There `Nx.LinAlg.eigh/2` runs a Jacobi
  # iteration whose default `eps: 1.0e-4` leaves the eigenvalues about three
  # digits short, and QDA divides by them and takes their log, so that error is
  # amplified rather than absorbed. Measured against LAPACK on the same f64
  # covariance, 1.0e-8 lands the eigenvalues within ~1.0e-7; tightening further
  # makes it worse, since the iteration then runs into `:max_iter`.
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

    # Every sample centered by the mean of its own class, so the loop below only
    # has to select rows rather than recentre them.
    centered = x - Nx.take(means, y, axis: 0)

    # One covariance per class, decomposed as we go. Looping keeps the working
    # set at O(num_samples * num_features); masking every class at once would
    # need a {num_classes, num_samples, num_features} intermediate instead.
    #
    # The decomposition also has to happen one class at a time. On Nx 0.9.x,
    # `Nx.LinAlg.eigh/2` over a stack of matrices decomposes only the first one
    # under EXLA and returns zeros for the rest. Fixed in Nx 0.13, but Scholar
    # still supports 0.9.
    scalings = Nx.broadcast(Nx.tensor(0, type: Nx.type(x)), {num_classes, num_features})

    rotations =
      Nx.broadcast(Nx.tensor(0, type: Nx.type(x)), {num_classes, num_features, num_features})

    {scalings, rotations, _} =
      while {scalings, rotations, {centered, one_hot, class_count, k = Nx.u32(0)}},
            k < num_classes do
        # The mask is one or zero, so masking a single side of the product is
        # enough to drop the rows belonging to the other classes.
        in_class = Nx.new_axis(one_hot[[.., k]], 1)
        covariance = Nx.dot(centered * in_class, [0], centered, [0]) / (class_count[k] - 1)

        {eigenvalues, eigenvectors} = Nx.LinAlg.eigh(covariance, eps: @eigh_eps)

        scalings = Nx.put_slice(scalings, [k, 0], Nx.new_axis(eigenvalues, 0))
        rotations = Nx.put_slice(rotations, [k, 0, 0], Nx.new_axis(eigenvectors, 0))

        {scalings, rotations, {centered, one_hot, class_count, k + 1}}
      end

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

    # Rotating `x - mean` is the same as rotating each separately, and doing it
    # separately keeps one {num_samples, num_classes, num_features} tensor around
    # instead of two.
    rotated =
      Nx.dot(x, [1], rotations, [1]) -
        Nx.new_axis(Nx.dot(means, [1], [0], rotations, [1], [0]), 0)

    mahalanobis = Nx.sum(rotated ** 2 / Nx.new_axis(scalings, 0), axes: [2])
    log_det = Nx.sum(Nx.log(scalings), axes: [1])

    -0.5 * (mahalanobis + Nx.new_axis(log_det, 0)) + Nx.new_axis(Nx.log(priors), 0)
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
