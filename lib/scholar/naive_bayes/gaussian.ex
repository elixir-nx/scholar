defmodule Scholar.NaiveBayes.Gaussian do
  @moduledoc ~S"""
  Gaussian Naive Bayes algorithm for classification.

  The likelihood of the features is assumed to be Gaussian:
  $$ P(x\_{i} | y) = \frac{1}{\sqrt{2\pi\sigma\_{y}^{2}}} \exp \left(-\frac{(x\_{i} - \mu\_{y})^2}{2\sigma\_{y}^{2}}\right) $$

  The parameters $\sigma\_{y}$ and $\mu\_{y}$ are estimated using maximum likelihood.

  Time complexity is $O(K * N * C)$ where $N$ is the number of samples and $K$ is the number of features,
  and $C$ is the number of classes.

  Reference:

  * [1] - [Detailed explanation of algorithm used to update feature means and variance online by Chan, Golub, and LaVeque](http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf)
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container,
           containers: [:theta, :var, :class_count, :class_priors, :classes, :epsilon]}
  defstruct [:theta, :var, :class_count, :class_priors, :classes, :epsilon]

  opts_schema = [
    var_smoothing: [
      type: :float,
      default: 1.0e-9,
      doc: ~S"""
      Portion of the largest variance of all features that is added to
      variances for calculation stability.
      """
    ],
    priors: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: ~S"""
      Prior probabilities of the classes. If specified, the priors are not
      adjusted according to the data. We assume that priors are correct and
      sum(priors) == 1.
      """
    ],
    sample_weights: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: ~S"""
      List of `n_samples` elements.

      A list of 1.0 values is used if none is given.
      """
    ],
    num_classes: [
      type: :pos_integer,
      required: true,
      doc: ~S"""
      Number of different classes used in training.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Gaussian Naive Bayes.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:theta` - mean of each feature per class.

    * `:var` - Variance of each feature per class.

    * `:class_count` - number of training samples observed in each class.

    * `:class_priors` - probability of each class.

    * `:classes` - class labels known to the classifier.

    * `:epsilon` - absolute additive value to variances.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> Scholar.NaiveBayes.Gaussian.fit(x, y, num_classes: 3)
      %Scholar.NaiveBayes.Gaussian{
        theta: Nx.tensor(
          [
            [6.0, 7.0, 8.0],
            [0.0, 1.0, 2.0],
            [6.0, 7.0, 8.0]
          ]
        ),
        var: Nx.tensor(
          [
            [1.1250000042650754e-8, 1.1250000042650754e-8, 1.1250000042650754e-8],
            [1.1250000042650754e-8, 1.1250000042650754e-8, 1.1250000042650754e-8],
            [9.0, 9.0, 9.0]
          ]
        ),
        class_count: Nx.tensor([1.0, 1.0, 2.0]),
        class_priors: Nx.tensor([0.25, 0.25, 0.5]),
        classes: Nx.tensor([0, 1, 2]),
        epsilon: Nx.tensor(1.1250000042650754e-8)
      }

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> Scholar.NaiveBayes.Gaussian.fit(x, y, num_classes: 3, sample_weights: [1, 6, 2, 3])
      %Scholar.NaiveBayes.Gaussian{
        theta: Nx.tensor(
          [
            [6.0, 7.0, 8.0],
            [0.0, 1.0, 2.0],
            [5.0, 6.0, 7.0]
          ]
        ),
        var: Nx.tensor(
          [
            [1.1250000042650754e-8, 1.1250000042650754e-8, 1.1250000042650754e-8],
            [1.1250000042650754e-8, 1.1250000042650754e-8, 1.1250000042650754e-8],
            [8.0, 8.0, 8.0]
          ]
          ),
        class_count: Nx.tensor([2.0, 1.0, 9.0]),
        class_priors: Nx.tensor([0.1666666716337204, 0.0833333358168602, 0.75]),
        classes: Nx.tensor([0, 1, 2]),
        epsilon: Nx.tensor(1.1250000042650754e-8)
      }
  """
  deftransform fit(x, y, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    opts =
      [
        priors_flag: opts[:priors] != nil
      ] ++
        opts

    x_type = to_float_type(x)

    {priors, opts} = Keyword.pop(opts, :priors, Nx.tensor(0.0, type: x_type))
    class_priors = Nx.tensor(priors)

    sample_weights_flag = Nx.tensor(opts[:sample_weights] != nil)
    {sample_weights, opts} = Keyword.pop(opts, :sample_weights, Nx.tensor(1.0, type: x_type))
    sample_weights = Nx.tensor(sample_weights, type: x_type)

    fit_n(x, y, sample_weights, class_priors, sample_weights_flag, opts)
  end

  @doc """
  Perform classification on an array of test vectors `x` using `model`.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Gaussian.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Gaussian.predict(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
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
  Return joint log probability estimates for the test vector `x` using `model`.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Gaussian.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Gaussian.predict_joint_log_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [-1822222336.0, -1822222208.0, -9.023576736450195],
          [-399999968.0, -5733332992.0, -7.245799541473389]
        ]
      >
  """
  defn predict_joint_log_probability(%__MODULE__{} = model, x) do
    check_input(model, x)
    joint_log_likelihood(model, x)
  end

  @doc """
  Return log-probability estimates for the test vector `x` using `model`.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Gaussian.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Gaussian.predict_log_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [-1822222336.0, -1822222208.0, 0.0],
          [-399999968.0, -5733332992.0, 0.0]
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
      |> Nx.new_axis(1)
      |> Nx.broadcast(jll)

    jll - log_proba_x
  end

  @doc """
  Return probability estimates for the test vector `x` using `model`.

  ## Examples

      iex> x = Nx.iota({4, 3})
      iex> y = Nx.tensor([1, 2, 0, 2])
      iex> model = Scholar.NaiveBayes.Gaussian.fit(x, y, num_classes: 3)
      iex> Scholar.NaiveBayes.Gaussian.predict_probability(model, Nx.tensor([[6, 2, 4], [8, 5, 9]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0]
        ]
      >
  """
  defn predict_probability(%__MODULE__{} = model, x) do
    Nx.exp(predict_log_probability(model, x))
  end

  defnp fit_n(x, y, sample_weights, class_priors, sample_weights_flag, opts) do
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

    eps = opts[:var_smoothing] * Nx.reduce_max(Nx.variance(x, axes: [0]))
    num_classes = opts[:num_classes]
    priors_flag = opts[:priors_flag]

    classes = Nx.iota({num_classes}) |> Nx.sort()

    class_priors =
      case Nx.shape(class_priors) do
        {} ->
          Nx.broadcast(class_priors, {num_classes})

        {^num_classes} ->
          class_priors

        _ ->
          raise ArgumentError,
                "number of priors must match number of classes. Number of priors: #{Nx.size(class_priors)} does not match number of classes: #{Nx.size(classes)}"
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

    {{final_theta, final_var, final_class_count}, _} =
      while {{theta = Nx.broadcast(Nx.tensor(0.0, type: x_type), {num_classes, num_features}),
              var = Nx.broadcast(Nx.tensor(0.0, type: x_type), {num_classes, num_features}),
              class_count = Nx.broadcast(Nx.tensor(0.0, type: x_type), {num_classes})},
             {i = 0, x, y, sample_weights, classes, sample_weights_flag}},
            i < Nx.size(classes) do
        y_i = classes[[i]]
        mask = y == y_i

        n_i = if sample_weights_flag, do: Nx.sum(mask * sample_weights), else: Nx.sum(mask)

        {new_mu, new_var, new_n} =
          update_mean_variance_init(
            theta[[i, ..]],
            var[[i, ..]],
            x,
            sample_weights,
            mask,
            sample_weights_flag
          )

        {new_mu, new_var} =
          if class_count[[i]] != 0,
            do:
              update_mean_variance(
                class_count[[i]],
                new_n,
                new_mu,
                new_var,
                theta[[i, ..]],
                var[[i, ..]]
              ),
            else: {new_mu, new_var}

        new_theta = Nx.put_slice(theta, [i, 0], Nx.broadcast(new_mu, {1, num_features}))
        new_var = Nx.put_slice(var, [i, 0], Nx.broadcast(new_var, {1, num_features}))
        class_count = Nx.indexed_add(class_count, Nx.reshape(i, {1, 1}), Nx.reshape(n_i, {1}))

        {{new_theta, new_var, class_count},
         {i + 1, x, y, sample_weights, classes, sample_weights_flag}}
      end

    final_var = final_var + eps

    class_priors =
      if priors_flag, do: class_priors, else: final_class_count / Nx.sum(final_class_count)

    %__MODULE__{
      theta: final_theta,
      var: final_var,
      class_count: final_class_count,
      class_priors: class_priors,
      classes: classes,
      epsilon: eps
    }
  end

  defnp update_mean_variance_init(mu, var, x, sample_weights, mask, sample_weights_flag) do
    {num_samples, _num_features} = Nx.shape(x)

    cond do
      num_samples == 0 ->
        {mu, var, 0}

      sample_weights_flag == false ->
        new_n = num_samples
        new_mu = mean_masked(x, mask)
        new_var = mean_masked((x - new_mu) ** 2, mask)
        {new_mu, new_var, new_n}

      true ->
        new_n = Nx.sum(sample_weights * mask)
        new_mu = mean_weighted_masked(x, mask, sample_weights)
        new_var = mean_weighted_masked((x - new_mu) ** 2, mask, sample_weights)
        {new_mu, new_var, new_n}
    end
  end

  defnp update_mean_variance(past_n, new_n, new_mu, new_var, mu, var) do
    total_n = past_n + new_n
    total_mu = (new_n * new_mu + past_n * mu) / total_n

    past_ssd = past_n * var
    new_ssd = new_n * new_var
    total_ssd = past_ssd + new_ssd + new_n * past_n / total_n * (mu - new_mu) ** 2
    total_var = total_ssd / total_n

    {total_mu, total_var}
  end

  defnp joint_log_likelihood(
          %__MODULE__{class_priors: class_priors, var: var, theta: theta},
          x
        ) do
    joint = Nx.log(class_priors)
    {num_classes, num_features} = Nx.shape(theta)
    {samples_x, _} = Nx.shape(x)

    n1 =
      (-0.5 * Nx.sum(Nx.log(2.0 * Nx.Constants.pi() * var), axes: [1]))
      |> Nx.new_axis(1)
      |> Nx.broadcast({num_classes, samples_x})

    broadcast_shape = {num_classes, samples_x, num_features}

    x_broadcast =
      Nx.new_axis(x, 0)
      |> Nx.broadcast(broadcast_shape)

    theta_broadcast =
      Nx.new_axis(theta, 1)
      |> Nx.broadcast(broadcast_shape)

    var_broadcast =
      Nx.new_axis(var, 1)
      |> Nx.broadcast(broadcast_shape)

    n2 = -0.5 * Nx.sum((x_broadcast - theta_broadcast) ** 2 / var_broadcast, axes: [2])

    Nx.transpose(n1 + n2) + joint
  end

  defnp mean_masked(t, mask) do
    broadcast_mask = mask |> Nx.new_axis(1) |> Nx.broadcast(t)
    Nx.sum(t * broadcast_mask, axes: [0]) / Nx.sum(broadcast_mask, axes: [0])
  end

  defnp mean_weighted_masked(t, mask, weights) do
    broadcast_mask = mask |> Nx.new_axis(1) |> Nx.broadcast(t)

    broadcast_weights = weights |> Nx.new_axis(1) |> Nx.broadcast(t)

    Nx.sum(t * broadcast_mask * broadcast_weights, axes: [0]) /
      Nx.sum(broadcast_mask * broadcast_weights, axes: [0])
  end

  defnp check_input(%__MODULE__{theta: theta}, x) do
    num_features = Nx.axis_size(theta, 1)
    x_num_features = Nx.axis_size(x, 1)

    if num_features != x_num_features do
      raise ArgumentError,
            "wrong input shape. Expected x to have the same second dimension as the data for fitting process, got: #{x_num_features} for x and #{num_features} for training data"
    end
  end
end
