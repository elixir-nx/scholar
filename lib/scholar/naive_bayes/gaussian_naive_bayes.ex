defmodule Scholar.NaiveBayes.Gaussian do
  @moduledoc """
  Univariate imputer for completing missing values with simple strategies.
  """
  import Nx.Defn

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
      type: {:list, {:custom, Scholar.Options, :positive_number, []}},
      doc: ~S"""
      Prior probabilities of the classes. If specified, the priors are not
      adjusted according to the data. We assume that priors are correct and
      sum(priors) == 1.
      """
    ],
    sample_weights: [
      type: {:list, {:or, [:float, :integer]}},
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

  For details on algorithm used to update feature means and variance online,
  see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

    http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Returns

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

    input_rank = Nx.rank(x)
    targets_rank = Nx.rank(y)

    if input_rank != 2 or targets_rank != 1 do
      raise ArgumentError,
            "Wrong input rank. Expected: 2 for x and 1 for y, got: #{inspect(input_rank)} for x and #{inspect(targets_rank)} for y"
    end

    {num_samples, _} = Nx.shape(x)
    {num_targets} = Nx.shape(y)

    if num_samples != num_targets do
      raise ArgumentError,
            "Wrong input shape. Expect x to have the same first dimension as y, got: #{inspect(num_samples)} for x and #{inspect(num_targets)} for y"
    end

    opts =
      opts ++
        [
          sample_weights_flag: if(opts[:sample_weights] != nil, do: true, else: false),
          priors_flag: if(opts[:priors] != nil, do: true, else: false)
        ]

    sample_weights =
      case opts[:sample_weights_flag] do
        false -> Nx.broadcast(1.0, {num_samples})
        _ -> Nx.tensor(opts[:sample_weights])
      end

    priors = opts[:priors]

    classes = Nx.iota({opts[:num_classes]})

    class_priors =
      case opts[:priors_flag] do
        false ->
          Nx.broadcast(0.0, Nx.shape(classes))

        _ ->
          priors = Nx.tensor(priors)

          if Nx.size(priors) != Nx.size(classes) do
            raise ArgumentError,
                  "Number of priors must match number of classes. Number of priors: #{inspect(Nx.size(priors))} does not match number of classes: #{inspect(Nx.size(classes))}"
          else
            priors
          end
      end

    fit_n(x, y, sample_weights, class_priors, classes, opts)
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
      |> Nx.reshape({:auto, 1})
      |> Nx.broadcast(Nx.shape(jll))

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

  defnp fit_n(x, y, sample_weights, class_priors, classes, opts \\ []) do
    eps = opts[:var_smoothing] * Nx.reduce_max(Nx.variance(x, axes: [0]))
    {_num_samples, num_features} = Nx.shape(x)
    sample_weights_flag = Nx.tensor(opts[:sample_weights_flag])
    priors_flag = opts[:priors_flag]
    classes = Nx.sort(classes)
    {num_classes} = Nx.shape(classes)

    {_, _, _, _, _, _, final_theta, final_var, final_class_count} =
      while {i = 0, x, y, sample_weights, classes, sample_weights_flag,
             theta = Nx.broadcast(0.0, {num_classes, num_features}),
             var = Nx.broadcast(0.0, {num_classes, num_features}),
             class_count = Nx.broadcast(0.0, {num_classes})},
            i < Nx.size(classes) do
        y_i = classes[[i]]
        mask = y == y_i

        n_i =
          if sample_weights_flag do
            Nx.sum(mask * sample_weights)
          else
            Nx.sum(mask)
          end

        {new_mu, new_var, new_n} =
          update_mean_variance_init(
            theta[[i, 0..-1//1]],
            var[[i, 0..-1//1]],
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
                theta[[i, 0..-1//1]],
                var[[i, 0..-1//1]]
              ),
            else: {new_mu, new_var}

        new_theta = Nx.put_slice(theta, [i, 0], Nx.broadcast(new_mu, {1, num_features}))
        new_var = Nx.put_slice(var, [i, 0], Nx.broadcast(new_var, {1, num_features}))
        class_count = Nx.indexed_add(class_count, Nx.reshape(i, {1, 1}), Nx.reshape(n_i, {1}))

        {i + 1, x, y, sample_weights, classes, sample_weights_flag, new_theta, new_var,
         class_count}
      end

    final_var = final_var + eps

    class_priors =
      case priors_flag do
        false -> final_class_count / Nx.sum(final_class_count)
        _ -> class_priors
      end

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

    if num_samples == 0 do
      {mu, var, 0}
    else
      case sample_weights_flag do
        false ->
          new_n = num_samples
          new_mu = mean_masked(x, mask)
          new_var = mean_masked((x - new_mu) ** 2, mask)
          {new_mu, new_var, new_n}

        _ ->
          new_n = Nx.sum(sample_weights * mask)
          new_mu = mean_weighted_masked(x, mask, sample_weights)
          new_var = mean_weighted_masked((x - new_mu) ** 2, mask, sample_weights)
          {new_mu, new_var, new_n}
      end
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
    pi = 3.14159265359
    joint = Nx.log(class_priors)
    {num_classes, num_features} = Nx.shape(theta)
    {samples_x, _features_x} = Nx.shape(x)

    n1 =
      (-0.5 * Nx.sum(Nx.log(2.0 * pi * var), axes: [1]))
      |> Nx.reshape({:auto, 1})
      |> Nx.broadcast({num_classes, samples_x})

    x_broadcast =
      Nx.reshape(x, {1, samples_x, num_features})
      |> Nx.broadcast({num_classes, samples_x, num_features})

    theta_broadcast =
      Nx.reshape(theta, {num_classes, 1, num_features})
      |> Nx.broadcast({num_classes, samples_x, num_features})

    var_broadcast =
      Nx.reshape(var, {num_classes, 1, num_features})
      |> Nx.broadcast({num_classes, samples_x, num_features})

    n2 = -0.5 * Nx.sum((x_broadcast - theta_broadcast) ** 2 / var_broadcast, axes: [2])

    (n1 + n2) |> Nx.transpose() |> Nx.add(joint)
  end

  defnp mean_masked(t, mask) do
    {num_samples, num_features} = Nx.shape(t)

    broadcast_mask =
      mask |> Nx.reshape({num_samples, 1}) |> Nx.broadcast({num_samples, num_features})

    Nx.sum(t * broadcast_mask, axes: [0]) / Nx.sum(broadcast_mask, axes: [0])
  end

  defnp mean_weighted_masked(t, mask, weights) do
    {num_samples, num_features} = Nx.shape(t)

    broadcast_mask =
      mask |> Nx.reshape({num_samples, 1}) |> Nx.broadcast({num_samples, num_features})

    broadcast_weights =
      weights |> Nx.reshape({num_samples, 1}) |> Nx.broadcast({num_samples, num_features})

    Nx.sum(t * broadcast_mask * broadcast_weights, axes: [0]) /
      Nx.sum(broadcast_mask * broadcast_weights, axes: [0])
  end

  defnp check_input(%__MODULE__{theta: theta}, x) do
    {_, num_features} = Nx.shape(theta)
    {_, x_num_features} = Nx.shape(x)

    if num_features != x_num_features do
      raise ArgumentError,
            "Wrong input shape. Expect x to have the same second dimension as the data for fitting process, got: #{inspect(x_num_features)} for x and #{inspect(num_features)} for training data"
    end
  end
end
