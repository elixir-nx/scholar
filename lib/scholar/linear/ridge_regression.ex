defmodule Scholar.Linear.RidgeRegression do
  @moduledoc ~S"""
  Linear least squares with $L_2$ regularization.

  Minimizes the objective function:
  $$
  ||y - Xw||^2\_2 + \alpha||w||^2\_2
  $$

  Where:
  * $X$ is an input data

  * $y$ is an input target

  * $w$ is the model weights matrix

  * $\alpha$ is the parameter that controls the level of regularization

  Time complexity is $O(N^2)$ for `:cholesky` solver and $O((N^2) * (K + N))$ for `:svd` solver,
  where $N$ is the number of observations and $K$ is the number of features.
  """
  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Linear.LinearHelpers

  @derive {Nx.Container, containers: [:coefficients, :intercept]}
  defstruct [:coefficients, :intercept]

  opts = [
    sample_weights: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: """
      The weights for each observation. If not provided,
      all observations are assigned equal weight.
      """
    ],
    fit_intercept?: [
      type: :boolean,
      default: true,
      doc: """
      If set to `true`, a model will fit the intercept. Otherwise,
      the intercept is set to `0.0`. The intercept is an independent term
      in a linear model. Specifically, it is the expected mean value
      of targets for a zero-vector on input.
      """
    ],
    solver: [
      type: {:in, [:svd, :cholesky]},
      default: :svd,
      doc: """
      Solver to use in the computational routines:

      * `:svd` - Uses a Singular Value Decomposition of A to compute the Ridge coefficients.
      In particular, it is more stable for singular matrices than `:cholesky` at the cost of being slower.

      * `:cholesky` - Uses the standard `Nx.LinAlg.solve` function to obtain a closed-form solution.
      """
    ],
    alpha: [
      type:
        {:or,
         [
           {:custom, Scholar.Options, :non_negative_number, []},
           {:list, {:custom, Scholar.Options, :non_negative_number, []}},
           {:custom, Scholar.Options, :weights, []}
         ]},
      default: 1.0,
      doc: ~S"""
      Constant that multiplies the $L_2$ term, controlling regularization strength.
      `:alpha` must be a non-negative float i.e. in [0, inf).

      If `:alpha` is set to 0.0 the objective is the ordinary least squares regression.
      In this case, for numerical reasons, you should use `Scholar.Linear.LinearRegression` instead.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a Ridge regression model for sample inputs `x` and
  sample targets `y`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:coefficients` - Estimated coefficients for the linear regression problem.

    * `:intercept` - Independent term in the linear model.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> Scholar.Linear.RidgeRegression.fit(x, y)
      %Scholar.Linear.RidgeRegression{
        coefficients: Nx.tensor(
          [-0.4237867593765259, -0.6891377568244934]
        ),
        intercept: Nx.tensor(
          5.6569366455078125
        )
      }
  """
  deftransform fit(x, y, opts \\ []) do
    {n_samples, _} = Nx.shape(x)
    y = LinearHelpers.flatten_column_vector(y, n_samples)

    opts = NimbleOptions.validate!(opts, @opts_schema)

    sample_weights? = opts[:sample_weights] != nil
    kernel_cholesky = opts[:solver] == :cholesky and Nx.axis_size(x, 0) < Nx.axis_size(x, 1)

    opts =
      [
        sample_weights_flag: sample_weights?,
        rescale_flag: sample_weights? and not kernel_cholesky
      ] ++
        opts

    x_type = to_float_type(x)

    sample_weights = LinearHelpers.build_sample_weights(x, opts)

    {alpha, opts} = Keyword.pop!(opts, :alpha)
    alpha = Nx.tensor(alpha, type: x_type) |> Nx.flatten()
    num_targets = if Nx.rank(y) == 1, do: 1, else: Nx.axis_size(y, 1)

    if Nx.size(alpha) not in [0, 1, num_targets] do
      raise ArgumentError,
            "expected number of targets be the same as number of penalties, got: #{inspect(num_targets)} != #{inspect(Nx.size(alpha))}"
    end

    fit_n(x, y, sample_weights, alpha, opts)
  end

  defnp fit_n(a, b, sample_weights, alpha, opts) do
    a = to_float(a)
    b = to_float(b)

    flatten? = Nx.rank(b) == 1
    num_targets = if flatten?, do: 1, else: Nx.axis_size(b, 1)

    alpha =
      if Nx.size(alpha) == 1 and num_targets > 1,
        do: Nx.broadcast(alpha, {num_targets}),
        else: alpha

    {a_offset, b_offset} =
      if opts[:fit_intercept?] do
        LinearHelpers.preprocess_data(a, b, sample_weights, opts)
      else
        a_offset_shape = Nx.axis_size(a, 1)
        b_reshaped = if Nx.rank(b) > 1, do: b, else: Nx.reshape(b, {:auto, 1})
        b_offset_shape = Nx.axis_size(b_reshaped, 1)

        {Nx.broadcast(Nx.tensor(0.0, type: Nx.type(a)), {a_offset_shape}),
         Nx.broadcast(Nx.tensor(0.0, type: Nx.type(b)), {b_offset_shape})}
      end

    {a, b} = {a - a_offset, b - b_offset}
    {num_samples, num_features} = Nx.shape(a)

    {a, b} =
      if opts[:rescale_flag] do
        LinearHelpers.rescale(a, b, sample_weights)
      else
        {a, b}
      end

    b = if Nx.rank(b) == 1, do: Nx.reshape(b, {:auto, 1}), else: b

    coeff =
      case opts[:solver] do
        :cholesky ->
          if num_samples >= num_features do
            solve_cholesky(a, b, alpha)
          else
            kernel = Nx.dot(a, [1], a, [1])
            dual_coeff = solve_cholesky_kernel(kernel, b, alpha, sample_weights, opts)
            Nx.dot(dual_coeff, [0], a, [0])
          end

        _ ->
          solve_svd(a, b, alpha)
      end

    coeff = if flatten?, do: Nx.flatten(coeff), else: coeff
    intercept = LinearHelpers.set_intercept(coeff, a_offset, b_offset, opts[:fit_intercept?])
    %__MODULE__{coefficients: coeff, intercept: intercept}
  end

  @doc """
  Makes predictions with the given `model` on input `x`.

  Output predictions have shape `{n_samples}` when train target is shaped either `{n_samples}` or `{n_samples, 1}`.        
  Otherwise, predictions match train target shape.  

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> model = Scholar.Linear.RidgeRegression.fit(x, y)
      iex> Scholar.Linear.RidgeRegression.predict(model, Nx.tensor([[2.0, 1.0]]))
      Nx.tensor(
        [4.120225429534912]
      )
  """
  defn predict(%__MODULE__{coefficients: coeff, intercept: intercept} = _model, x) do
    original_rank = Nx.rank(coeff)
    coeff = if original_rank == 1, do: Nx.new_axis(coeff, 0), else: coeff
    res = Nx.dot(x, [-1], coeff, [-1]) + intercept
    if original_rank <= 1, do: Nx.squeeze(res, axes: [1]), else: res
  end

  defnp solve_cholesky_kernel(kernel, b, alpha, sample_weights, opts) do
    num_samples = Nx.axis_size(kernel, 0)
    num_targets = Nx.axis_size(b, 1)
    one_alpha = Nx.all(alpha[[0]] == alpha)
    sample_weights = Nx.sqrt(sample_weights)

    {kernel, b} =
      if opts[:sample_weights_flag] do
        b = b * Nx.new_axis(sample_weights, 1)
        {kernel * Nx.outer(sample_weights, sample_weights), b}
      else
        {kernel, b}
      end

    if one_alpha do
      kernel = kernel + Nx.eye(num_samples) * alpha[[0]]
      dual_coeff = Nx.LinAlg.solve(kernel, b)

      if opts[:sample_weights_flag],
        do: dual_coeff * Nx.new_axis(sample_weights, 1),
        else: dual_coeff
    else
      b = Nx.transpose(b) |> Nx.new_axis(-1)

      broadcast_shape = {num_targets, num_samples, num_samples}
      kernel = Nx.new_axis(kernel, 0) |> Nx.broadcast(broadcast_shape)

      reg =
        Nx.new_axis(Nx.eye(num_samples), 0)
        |> Nx.broadcast(broadcast_shape)

      reg = reg * Nx.reshape(alpha, {:auto, 1, 1})
      kernel = kernel + reg
      dual_coeff = Nx.LinAlg.solve(kernel, b) |> Nx.squeeze(axes: [-1])

      dual_coeff =
        if opts[:sample_weights_flag],
          do: dual_coeff * Nx.new_axis(sample_weights, 0),
          else: dual_coeff

      Nx.transpose(dual_coeff)
    end
  end

  defnp solve_cholesky(a, b, alpha) do
    num_features = Nx.axis_size(a, 1)
    num_targets = Nx.axis_size(b, 1)

    kernel = Nx.dot(a, [0], a, [0])

    ab = Nx.dot(a, [0], b, [0])

    one_alpha = Nx.all(alpha[[0]] == alpha)

    if one_alpha do
      kernel = kernel + Nx.eye(num_features) * alpha[[0]]
      Nx.LinAlg.solve(kernel, ab) |> Nx.transpose()
    else
      target = Nx.transpose(ab)

      target = Nx.new_axis(target, -1)

      broadcast_shape = {num_targets, num_features, num_features}
      kernel = Nx.new_axis(kernel, 0) |> Nx.broadcast(broadcast_shape)

      reg =
        Nx.new_axis(Nx.eye(num_features), 0)
        |> Nx.broadcast(broadcast_shape)

      reg = reg * Nx.reshape(alpha, {:auto, 1, 1})
      kernel = kernel + reg
      Nx.LinAlg.solve(kernel, target) |> Nx.squeeze(axes: [-1])
    end
  end

  defnp solve_svd(a, b, alpha) do
    {u, s, vt} = Nx.LinAlg.svd(a, full_matrices?: false)
    s_size = Nx.size(s)
    alpha_size = Nx.size(alpha)
    broadcast_size = {s_size, alpha_size}
    idx = (s > 1.0e-15) |> Nx.new_axis(1) |> Nx.broadcast(broadcast_size)
    s = Nx.new_axis(s, 1) |> Nx.broadcast(broadcast_size)
    alpha = Nx.new_axis(alpha, 0) |> Nx.broadcast(broadcast_size)
    d = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(a)), broadcast_size)
    d = Nx.select(idx, s / (s ** 2 + alpha), d)
    uty = Nx.dot(u, [0], b, [0])
    d_uty = d * uty
    Nx.dot(d_uty, [0], vt, [0])
  end
end
