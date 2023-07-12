defmodule Scholar.Cluster.GaussianMixture do
	@moduledoc """
  Gaussian Mixture Model.

  Gaussian Mixture model is a probabilistic model that assumes every data point is generated
	by randomly choosing one of the several fixed Gaussian distributions and then sampling from it.
	Its parameters are estimated using the Expectation-Maximization (EM) algorithm, which is an
	iterative algorithm alternating between the two steps: the E-step which computes the
	expectation of the Gaussian assignment for each data point x and the M-step which updates the
	parameters to maximize the expectations found in E-step. While every iteration of the algorithm
	is guaranteed to improve the log-likelihood, the final result depends on the initial values of
	the parameters and the entire procedure should be repeated several times.

  References:

  * [1] - https://cs229.stanford.edu/notes2020spring/cs229-notes7b.pdf
  """

  import Nx.Defn
  import Scholar.Shared
  require Nx

	@derive {Nx.Container, containers: [:weights, :means, :covariances, :precisions_cholesky]}
	defstruct [:weights, :means, :covariances, :precisions_cholesky]

  opts = [
  	num_gaussians: [
  		required: true,
  		type: :pos_integer,
  		doc: "The number of Gaussian components in the mixture."
  	],
    num_runs: [
      type: :pos_integer,
      default: 1,
      doc: "The number of initializations to perform."
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
    reg_covar: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-6,
      doc: """
			The non-negative number that is added to each element of the diagonal
			of the covariance matrix to ensure it is positive.
			"""
    ],
    key: [
    	type: {:custom, Scholar.Options, :key, []},
      doc: """
      Determines random number generation for parameter initialization.
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

    * `:weights` - ..

    * `:means` - Means of the Gaussian components.

    * `:covariances` - Covariance matrices of the Gaussian components.

    * `:precisions_cholesky` - Cholesky decomposition of the precision matrices
			(inverse of covariance). This is useful for the inference.

  ## Examples

      iex> key = Nx.Random.key(12)
      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> Scholar.Cluster.KMeans.fit(x, num_clusters: 2, key: key)
      %Scholar.Cluster.KMeans{
        num_iterations: Nx.tensor(
          2
        ),
        clusters: Nx.tensor(
          [
            [1.0, 2.5],
            [2.0, 4.5]
          ]
        ),
        inertia: Nx.tensor(
          1.0
        ),
        labels: Nx.tensor(
          [0, 1, 0, 1]
        )
      }
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
    reg_covar = opts[:reg_covar]

		{best_params, _} =
      while {
				best_params = init_params(key, x, num_gaussians: num_gaussians, reg_covar: reg_covar),
				{
					run = 1, max_lower_bound = Nx.Constants.neg_infinity,
					x, max_iter, tol, reg_covar
				}
      },
			run <= num_runs do

				{{lower_bound, params}, _} =
          while {
						{
							lower_bound = Nx.Constants.neg_infinity,
							params = init_params(key, x, num_gaussians: num_gaussians, reg_covar: reg_covar),
						},
						{
            	iter = 1,
            	converged = Nx.tensor(false),
							x, max_iter, tol, reg_covar
						}
          },
					iter <= max_iter and not converged do

            prev_lower_bound = lower_bound
            {weights, means, _, precisions_cholesky} = params
            {log_prob_norm, log_resp} = e_step(x, weights, means, precisions_cholesky)
            params = m_step(x, log_resp, reg_covar)
            lower_bound = log_prob_norm
            change = lower_bound - prev_lower_bound
            converged = Nx.abs(change) < tol
						{
							{lower_bound, params},
							{iter + 1, converged, x, max_iter, tol, reg_covar}
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
					{run + 1, max_lower_bound, x, max_iter, tol, reg_covar}
				}

      end

    # %__MODULE__{
    #   weights: weights,
    #   means: means,
    #   covariances: covariances,
    #   precisions_cholesky: precisions_cholesky
    # }

    best_params
  end

  defnp init_params(key, x, opts) do
    type = Nx.type(x)
    {num_samples, _} = Nx.shape(x)
    num_gaussians = opts[:num_gaussians]
    reg_covar = opts[:reg_covar]
  	# k_means_model = Scholar.Cluster.KMeans.fit(x, num_clusters: num_gaussians, num_runs: 1, key: key)
    k_means_model = Scholar.Cluster.KMeans.fit(x, num_clusters: num_gaussians, num_runs: 1) # add key here
  	labels = k_means_model.labels
  	resp = Nx.take(Nx.eye(num_gaussians, type: type), labels)
  	{weights, means, covariances, precisions_cholesky} = estimate_gaussian_parameters(x, resp, reg_covar)
		weights = weights / num_samples
		{weights, means, covariances, precisions_cholesky}
  end

	defnp e_step(x, weights, means, precisions_cholesky) do
		{log_prob_norm, log_resp} = estimate_log_prob_resp(x, weights, means, precisions_cholesky)
  	{Nx.mean(log_prob_norm), log_resp}
	end

	defnp m_step(x, log_resp, reg_covar) do
		{weights, means, covariances, precisions_cholesky} =
			estimate_gaussian_parameters(x, Nx.exp(log_resp), reg_covar)
		weights = weights / Nx.sum(weights)
		{weights, means, covariances, precisions_cholesky}
	end

  defnp estimate_gaussian_parameters(x, resp, reg_covar) do
  	nk = Nx.sum(resp, axes: [0]) # consider adding a small number here as in scikit_learn
  	means = Nx.dot(Nx.transpose(resp), x) / nk
  	covariances = estimate_gaussian_covariances(x, resp, nk, means, reg_covar)
  	precisions_cholesky = compute_precisions_cholesky(covariances)
  	{nk, means, covariances, precisions_cholesky}
  end

  defnp estimate_gaussian_covariances(x, resp, nk, means, reg_covar) do
    type = Nx.type(x)
  	{num_gaussians, num_features} = Nx.shape(means)
		{covariances, _} =
  		while {
				covariances =
				  Nx.tensor(0.0, type: type)
					|> Nx.broadcast({num_gaussians, num_features, num_features}),
				{k = 0, x, resp, nk, means, reg_covar}
			},
			k < num_gaussians do
  			diff = x - means[k]
  			covariance = Nx.dot(resp[[.., k]] * Nx.transpose(diff), diff) / nk[k]
  			covariance = Nx.put_diagonal(covariance, Nx.take_diagonal(covariance) + reg_covar)
  			{
					Nx.put_slice(covariances, [k, 0, 0], Nx.new_axis(covariance, 0)),
					{k + 1, x, resp, nk, means, reg_covar}
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
					|> Nx.transpose
  			{
					Nx.put_slice(precisions_cholesky, [k, 0, 0], Nx.new_axis(precision_cholesky, 0)),
					{k + 1, covariances}
				}
  		end
  	precisions_cholesky
  end

	defnp estimate_log_prob_resp(x, weights, means, precisions_cholesky) do
		weighted_log_prob = estimate_weighted_log_prob(x, weights, means, precisions_cholesky)
		log_prob_norm =
			weighted_log_prob
			|> Nx.exp
			|> Nx.sum(axes: [1])
			|> Nx.log
		log_resp = weighted_log_prob - Nx.new_axis(log_prob_norm, 1)
		{log_prob_norm, log_resp}
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
				slice = Nx.new_axis(Nx.sum(Nx.multiply(y, y), axes: [1]), 0)
				{
					Nx.put_slice(log_prob, [k, 0], slice),
					{k + 1, x, means, precisions_cholesky}
				}
			end
		-0.5 * (num_features * Nx.log(2 * Nx.Constants.pi) + Nx.transpose(log_prob)) + log_det
		# log_prob
		# 	|> Nx.transpose
		# 	|> Nx.add(num_features * Nx.log(2 * Nx.Constants.pi))
		# 	|> Nx.multiply(-0.5)
		# 	|> Nx.add(log_det)
	end

	defnp compute_log_det_cholesky(precisions_cholesky) do
		precisions_cholesky
		|> Nx.take_diagonal
		|> Nx.log
		|> Nx.sum(axes: [1])
	end

	@doc """
  Makes predictions with the given `model` on inputs `x`.

  ## Return Values

    It returns a tensor with clusters corresponding to the input.

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> model = Scholar.Cluster.KMeans.fit(x, num_clusters: 2, key: key)
      iex> Scholar.Cluster.KMeans.predict(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      Nx.tensor(
        [1, 0]
      )
  """
	defn predict(
		%__MODULE__{weights: weights, means: means, precisions_cholesky: precisions_cholesky}, x
	) do
    assert_same_shape!(x[0], means[0])
    estimate_weighted_log_prob(x, weights, means, precisions_cholesky) |> Nx.argmax(axis: 1)
  end

	@doc """
	"""
	defn predict_prob(
		%__MODULE__{weights: weights, means: means, precisions_cholesky: precisions_cholesky}, x
	) do
	  assert_same_shape!(x[0], means[0])
		{_, log_resp} = estimate_log_prob_resp(x, weights, means, precisions_cholesky)
		Nx.exp(log_resp)
	end

end
