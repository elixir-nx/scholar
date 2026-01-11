defmodule Scholar.Cluster.GaussianMixture do
  @moduledoc """
  Gaussian Mixture Model.

  Gaussian Mixture Model is a probabilistic model that assumes every data point is generated
  by choosing one of several fixed Gaussian distributions and then sampling from it.
  Its parameters are estimated using the Expectation-Maximization (EM) algorithm, which is an
  iterative algorithm alternating between the two steps: the E-step which computes the
  expectation of the Gaussian assignment for each data point x and the M-step which updates the
  parameters to maximize the expectations found in E-step. While every iteration of the algorithm
  is guaranteed to improve the log-likelihood, the final result depends on the initial values of
  the parameters. Thus the procedure consists of repeating the algorithm several times
  and taking the best obtained result.

  Time complexity is $O(NKD^3)$ for $N$ data points, $K$ Gaussian components and $D$ dimensions

  References:

  * [1] - Mixtures of Gaussians and the EM algorithm https://cs229.stanford.edu/notes2020spring/cs229-notes7b.pdf
  * [2] - Density Estimation with Gaussian Mixture Models https://mml-book.github.io/book/mml-book.pdf Chapter 11
  """

  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:weights, :means, :covariances, :precisions_cholesky]}
  defstruct [:weights, :means, :covariances, :precisions_cholesky]

  opts = [
    num_gaussians: [
      required: true,
      type: :pos_integer,
      doc: "The number of Gaussian distributions in the mixture."
    ],
    num_runs: [
      type: :pos_integer,
      default: 1,
      doc: "The number of times to initialize parameters and run the entire EM algorithm."
    ],
    max_iter: [
      type: :pos_integer,
      default: 100,
      doc: "The number of EM iterations to perform."
    ],
    tol: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-3,
      doc: "The convergence threshold."
    ],
    covariance_regularization_eps: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-6,
      doc: """
      The non-negative number that is added to each element of the diagonal
      of the covariance matrix to ensure it is positive. Usually a small number.
      """
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Used for random number generation in parameter initialization.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a Gaussian Mixture Model for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:weights` - The fractions of data sampled from each Gaussian, respectively.

    * `:means` - Means of the Gaussian components.

    * `:covariances` - Covariance matrices of the Gaussian components.

    * `:precisions_cholesky` - Cholesky decomposition of the precision matrices
  	(inverses of covariances). This is useful for the model inference.

  ## Examples

      iex> key = Nx.Random.key(12)
      iex> x = Nx.tensor([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
      iex> Scholar.Cluster.GaussianMixture.fit(x, num_gaussians: 2, key: key).means
      Nx.tensor(
        [
          [1.0, 2.0],
          [10.0, 2.0]
        ]
      )
  """
  deftransform fit(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected input tensor to have shape {n_samples, n_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}\
            """
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)

    num_samples = Nx.axis_size(x, 0)

    unless opts[:num_gaussians] <= num_samples do
      raise ArgumentError,
            """
            invalid value for :num_gaussians option: \
            expected positive integer between 1 and #{inspect(num_samples)}, \
            got: #{inspect(opts[:num_gaussians])}\
            """
    end

    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_n(x, key, opts)
  end

  defnp fit_n(x, key, opts) do
    x = to_float(x)
    num_gaussians = opts[:num_gaussians]
    num_runs = opts[:num_runs]
    max_iter = opts[:max_iter]
    tol = opts[:tol]
    covariance_regularization_eps = opts[:covariance_regularization_eps]

    {best_params, _} =
      while {
              best_params =
                init_params(key, x,
                  num_gaussians: num_gaussians,
                  covariance_regularization_eps: covariance_regularization_eps
                ),
              {
                run = 1,
                max_lower_bound = Nx.Constants.neg_infinity(),
                x,
                key,
                max_iter,
                tol,
                covariance_regularization_eps
              }
            },
            run <= num_runs do
        {{lower_bound, params}, _} =
          while {
                  {
                    lower_bound = Nx.Constants.neg_infinity(),
                    params =
                      init_params(key, x,
                        num_gaussians: num_gaussians,
                        covariance_regularization_eps: covariance_regularization_eps
                      )
                  },
                  {
                    iter = 1,
                    converged = Nx.tensor(false),
                    x,
                    key,
                    max_iter,
                    tol,
                    covariance_regularization_eps
                  }
                },
                iter <= max_iter and not converged do
            prev_lower_bound = lower_bound
            {weights, means, _, precisions_cholesky} = params
            {log_prob_norm, log_responsibilities} = e_step(x, weights, means, precisions_cholesky)
            params = m_step(x, log_responsibilities, covariance_regularization_eps)
            lower_bound = log_prob_norm
            change = lower_bound - prev_lower_bound
            converged = iter > 1 and Nx.abs(change) < tol

            {
              {lower_bound, params},
              {iter + 1, converged, x, key, max_iter, tol, covariance_regularization_eps}
            }
          end

        {max_lower_bound, best_params} =
          if max_lower_bound < lower_bound do
            {lower_bound, params}
          else
            {max_lower_bound, best_params}
          end

        {
          best_params,
          {run + 1, max_lower_bound, x, key, max_iter, tol, covariance_regularization_eps}
        }
      end

    {weights, means, covariances, precisions_cholesky} = best_params

    %__MODULE__{
      weights: weights,
      means: means,
      covariances: covariances,
      precisions_cholesky: precisions_cholesky
    }
  end

  defnp init_params(key, x, opts) do
    type = Nx.type(x)
    {num_samples, _} = Nx.shape(x)
    num_gaussians = opts[:num_gaussians]
    covariance_regularization_eps = opts[:covariance_regularization_eps]

    k_means_model =
      Scholar.Cluster.KMeans.fit(x, num_clusters: num_gaussians, num_runs: 1, key: key)

    labels = k_means_model.labels
    responsibilities = Nx.take(Nx.eye(num_gaussians, type: type), labels)

    {weights, means, covariances, precisions_cholesky} =
      estimate_gaussian_parameters(x, responsibilities, covariance_regularization_eps)

    weights = weights / num_samples
    {weights, means, covariances, precisions_cholesky}
  end

  defnp e_step(x, weights, means, precisions_cholesky) do
    {log_prob_norm, log_responsibilities} =
      estimate_log_prob_responsibilities(x, weights, means, precisions_cholesky)

    {Nx.mean(log_prob_norm), log_responsibilities}
  end

  defnp m_step(x, log_responsibilities, covariance_regularization_eps) do
    {weights, means, covariances, precisions_cholesky} =
      estimate_gaussian_parameters(x, Nx.exp(log_responsibilities), covariance_regularization_eps)

    weights = weights / Nx.sum(weights)
    {weights, means, covariances, precisions_cholesky}
  end

  defnp estimate_gaussian_parameters(x, responsibilities, covariance_regularization_eps) do
    nk = Nx.sum(responsibilities, axes: [0])
    means = Nx.dot(responsibilities, [0], x / nk, [0])

    covariances =
      estimate_gaussian_covariances(x, responsibilities, nk, means, covariance_regularization_eps)

    precisions_cholesky = compute_precisions_cholesky(covariances)
    {nk, means, covariances, precisions_cholesky}
  end

  defnp estimate_gaussian_covariances(
          x,
          responsibilities,
          nk,
          means,
          covariance_regularization_eps
        ) do
    type = Nx.type(x)
    {num_gaussians, num_features} = Nx.shape(means)

    {covariances, _} =
      while {
              covariances =
                Nx.tensor(0.0, type: type)
                |> Nx.broadcast({num_gaussians, num_features, num_features}),
              {k = 0, x, responsibilities, nk, means, covariance_regularization_eps}
            },
            k < num_gaussians do
        diff = x - means[k]

        covariance =
          Nx.dot(diff * Nx.new_axis(responsibilities[[.., k]], 1), [0], diff / nk[k], [0])

        covariance =
          Nx.put_diagonal(
            covariance,
            Nx.take_diagonal(covariance) + covariance_regularization_eps
          )

        {
          Nx.put_slice(covariances, [k, 0, 0], Nx.new_axis(covariance, 0)),
          {k + 1, x, responsibilities, nk, means, covariance_regularization_eps}
        }
      end

    covariances
  end

  defnp compute_precisions_cholesky(covariances) do
    type = Nx.type(covariances)
    {num_gaussians, num_features, _} = Nx.shape(covariances)

    {precisions_cholesky, _} =
      while {
              precisions_cholesky =
                Nx.tensor(0.0, type: type)
                |> Nx.broadcast({num_gaussians, num_features, num_features}),
              {k = 0, covariances}
            },
            k < num_gaussians do
        covariance_cholesky = Nx.LinAlg.cholesky(covariances[k])

        precision_cholesky =
          covariance_cholesky
          |> Nx.LinAlg.triangular_solve(Nx.eye(num_features, type: type), lower: true)
          |> Nx.transpose()

        {
          Nx.put_slice(precisions_cholesky, [k, 0, 0], Nx.new_axis(precision_cholesky, 0)),
          {k + 1, covariances}
        }
      end

    precisions_cholesky
  end

  defnp estimate_log_prob_responsibilities(x, weights, means, precisions_cholesky) do
    weighted_log_prob = estimate_weighted_log_prob(x, weights, means, precisions_cholesky)
    log_prob_norm = logsumexp(weighted_log_prob)
    log_responsibilities = weighted_log_prob - Nx.new_axis(log_prob_norm, 1)
    {log_prob_norm, log_responsibilities}
  end

  defnp estimate_weighted_log_prob(x, weights, means, precisions_cholesky) do
    estimate_log_prob(x, means, precisions_cholesky) + Nx.log(weights)
  end

  defnp estimate_log_prob(x, means, precisions_cholesky) do
    type = Nx.type(x)
    {num_samples, num_features} = Nx.shape(x)
    {num_gaussians, _} = Nx.shape(means)
    log_det = compute_log_det_cholesky(precisions_cholesky)

    {log_prob, _} =
      while {
              log_prob =
                Nx.tensor(0.0, type: type)
                |> Nx.broadcast({num_gaussians, num_samples}),
              {k = 0, x, means, precisions_cholesky}
            },
            k < num_gaussians do
        mean = means[k]
        precision_cholesky = precisions_cholesky[k]
        y = Nx.dot(x, precision_cholesky) - Nx.dot(mean, precision_cholesky)
        slice = Nx.new_axis(Nx.sum(y * y, axes: [1]), 0)

        {
          Nx.put_slice(log_prob, [k, 0], slice),
          {k + 1, x, means, precisions_cholesky}
        }
      end

    -0.5 * (num_features * Nx.log(2 * Nx.Constants.pi()) + Nx.transpose(log_prob)) + log_det
  end

  defnp logsumexp(weighted_log_prob) do
    max = Nx.reduce_max(weighted_log_prob, axes: [1])

    weighted_log_prob
    |> Nx.subtract(Nx.new_axis(max, 1))
    |> Nx.exp()
    |> Nx.sum(axes: [1])
    |> Nx.log()
    |> Nx.add(max)
  end

  defnp compute_log_det_cholesky(precisions_cholesky) do
    precisions_cholesky
    |> Nx.take_diagonal()
    |> Nx.log()
    |> Nx.sum(axes: [1])
  end

  @doc """
  Makes predictions with the given `model` on inputs `x`.

  ## Return Values

    It returns a tensor with Gaussian assignments for every input point.

  ## Examples

      iex> key = Nx.Random.key(12)
      iex> x = Nx.tensor([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
      iex> model = Scholar.Cluster.GaussianMixture.fit(x, num_gaussians: 2, key: key)
      iex> Scholar.Cluster.GaussianMixture.predict(model, Nx.tensor([[8, 1], [2, 3]]))
      Nx.tensor(
        [1, 0]
      )
  """
  defn predict(
         %__MODULE__{weights: weights, means: means, precisions_cholesky: precisions_cholesky} =
           _model,
         x
       ) do
    assert_same_shape!(x[0], means[0])

    estimate_weighted_log_prob(to_float(x), weights, means, precisions_cholesky)
    |> Nx.argmax(axis: 1)
  end

  @doc """
  Makes predictions with the given `model` on inputs `x`.

  ## Return Values

    It returns a tensor probabilities of Gaussian assignments for every input point.

  ## Examples

      iex> key = Nx.Random.key(12)
      iex> x = Nx.tensor([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
      iex> model = Scholar.Cluster.GaussianMixture.fit(x, num_gaussians: 2, key: key)
      iex> Scholar.Cluster.GaussianMixture.predict_prob(model, Nx.tensor([[8, 1], [2, 3]]))
      Nx.tensor(
        [
          [0.0, 1.0],
          [1.0, 0.0]
        ]
      )
  """
  defn predict_prob(
         %__MODULE__{weights: weights, means: means, precisions_cholesky: precisions_cholesky} =
           _model,
         x
       ) do
    assert_same_shape!(x[0], means[0])

    {_, log_responsibilities} =
      estimate_log_prob_responsibilities(to_float(x), weights, means, precisions_cholesky)

    Nx.exp(log_responsibilities)
  end
end
