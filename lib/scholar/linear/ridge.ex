defmodule Scholar.Linear.Ridge do
  @moduledoc ~S"""
  Linear least squares with $L_2$ regularization.

  Minimizes the objective function:
  $$
  ||y - Xw||^2\_2 + \alpha||w||^2\_2
  $$

  Where:
  * $X$ is an input data

  * $y$ is an input target

  * $w$ is a model weights matrix

  * $\alpha$ ia a parameter that controls level of regularization
  """
  import Nx.Defn

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

      * `:svd` - Uses a Singular Value Decomposition of X to compute the Ridge coefficients.
      In particular more stable for singular matrices than `:cholesky` at the cost of being slower.

      * `:cholesky` - Uses the standard `Nx.LinAlg.solve` function to obtain a closed-form solution.
      """
    ],
    alpha: [
      type:
        {:or,
         [
           {:custom, Scholar.Options, :non_negative_number, []},
           {:list, {:custom, Scholar.Options, :non_negative_number, []}}
         ]},
      default: 1.0,
      doc: ~S"""
      Constant that multiplies the $L_2$ term, controlling regularization strength.
      `:alpha` must be a non-negative float i.e. in [0, inf).

      If `:alpha` is set to 0.0 than objective is ordinary least squares regression.
      For numerical reasons you should use `Scholar.Linear.LinearRegression` instead.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a ridge regression model for sample inputs `a` and
  sample targets `b`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:coefficients` - Estimated coefficients for the linear regression problem.

    * `:intercept` - Independent term in the linear model.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> Scholar.Linear.Ridge.fit(x, y)
      %Scholar.Linear.Ridge{
        coefficients: #Nx.Tensor<
          f32[2]
          [-0.42378732562065125, -0.6891375780105591]
        >,
        intercept: #Nx.Tensor<
          f32
          5.656937599182129
        >
      }
  """
  deftransform fit(a, b, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    sample_weights? = opts[:sample_weights] != nil
    kernel_cholesky = opts[:solver] == :cholesky and Nx.axis_size(a, 0) < Nx.axis_size(a, 1)

    opts =
      [
        sample_weights_flag: sample_weights?,
        rescale_flag: sample_weights? and not kernel_cholesky
      ] ++
        opts

    {sample_weights, opts} = Keyword.pop(opts, :sample_weights, 1.0)
    sample_weights = Nx.tensor(sample_weights)

    {alpha, opts} = Keyword.pop!(opts, :alpha)
    alpha = Nx.tensor(alpha) |> Nx.flatten()
    num_targets = if Nx.rank(b) == 1, do: 1, else: Nx.axis_size(b, 1)

    if Nx.size(alpha) not in [0, 1, num_targets] do
      raise ArgumentError,
            "expected number of targets be the same as number of penalties, got: #{inspect(num_targets)} != #{inspect(Nx.size(alpha))}"
    end

    fit_n(a, b, sample_weights, alpha, opts)
  end

  defnp fit_n(a, b, sample_weights, alpha, opts) do
    flatten? = Nx.rank(b) == 1
    num_targets = if Nx.rank(b) == 1, do: 1, else: Nx.axis_size(b, 1)

    alpha =
      if Nx.size(alpha) == 1 and num_targets > 1,
        do: Nx.broadcast(alpha, {num_targets}),
        else: alpha

    {a_offset, b_offset} =
      if opts[:fit_intercept?] do
        preprocess_data(a, b, sample_weights, opts)
      else
        {_, a_shape} = Nx.shape(a)
        b_reshaped = if Nx.rank(b) > 1, do: b, else: Nx.reshape(b, {:auto, 1})
        {_, b_shape} = Nx.shape(b_reshaped)
        {Nx.broadcast(0.0, {a_shape}), Nx.broadcast(0.0, {b_shape})}
      end

    {a, b} = {a - a_offset, b - b_offset}
    {num_samples, num_features} = Nx.shape(a)

    {a, b} =
      if opts[:rescale_flag] do
        rescale(a, b, sample_weights)
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
            Nx.dot(a, [0], dual_coeff, [0]) |> Nx.transpose()
          end

        _ ->
          solve_svd(a, b, alpha)
      end

    coeff = if flatten?, do: Nx.flatten(coeff), else: coeff
    intercept = set_intercept(coeff, a_offset, b_offset, opts[:fit_intercept?])
    %__MODULE__{coefficients: coeff, intercept: intercept}
  end

  @doc """
  Makes predictions with the given `model` on input `x`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> model = Scholar.Linear.Ridge.fit(x, y)
      iex> Scholar.Linear.Ridge.predict(model, Nx.tensor([[2.0, 1.0]]))
      #Nx.Tensor<
        f32[1]
        [4.120225429534912]
      >
  """
  defn predict(%__MODULE__{coefficients: coeff, intercept: intercept} = _model, x) do
    Nx.dot(x, coeff) + intercept
  end

  # Implements sample weighting by rescaling inputs and
  # targets by sqrt(sample_weight).
  defn rescale(x, y, sample_weights) do
    case Nx.shape(sample_weights) do
      {} = scalar ->
        scalar = Nx.sqrt(scalar)
        {scalar * x, scalar * y}

      _ ->
        scale = sample_weights |> Nx.sqrt() |> Nx.make_diagonal()
        {Nx.dot(scale, x), Nx.dot(scale, y)}
    end
  end

  defnp solve_cholesky_kernel(kernel, y, alpha, sample_weights, opts) do
    num_samples = Nx.axis_size(kernel, 0)
    num_targets = Nx.axis_size(y, 1)
    one_alpha = Nx.all(alpha[[0]] == alpha)
    sample_weights = Nx.sqrt(sample_weights)

    {kernel, y} =
      if opts[:sample_weights_flag] do
        y = y * Nx.new_axis(sample_weights, 1)
        {kernel * Nx.outer(sample_weights, sample_weights), y}
      end

    if one_alpha do
      kernel = kernel + Nx.eye(num_samples) * alpha[[0]]
      dual_coeff = Nx.LinAlg.solve(kernel, y)

      if opts[:sample_weights_flag],
        do: dual_coeff * Nx.new_axis(sample_weights, 1),
        else: dual_coeff
    else
      y = Nx.transpose(y)

      # {_, _, _, dual_coeff, _} =
      #   while {kernel, y, alpha, dual_coeff = Nx.broadcast(0.0, {num_targets, num_samples}),
      #          i = 0},
      #         i < num_targets do
      #     kernel = kernel + Nx.eye(num_samples) * alpha[[i]]
      #     dual_coeff_slice = Nx.LinAlg.solve(kernel, y[[i]]) |> Nx.new_axis(0)
      #     dual_coeff = Nx.put_slice(dual_coeff, [i, 0], dual_coeff_slice)
      #     kernel = kernel - Nx.eye(num_samples) * alpha[[i]]
      #     {kernel, y, alpha, dual_coeff, i + 1}
      #   end

      y = Nx.new_axis(y, -1)

      kernel = Nx.new_axis(kernel, 0) |> Nx.broadcast({num_targets, num_samples, num_samples})

      reg =
        Nx.new_axis(Nx.eye(num_samples), 0)
        |> Nx.broadcast({num_targets, num_samples, num_samples})

      reg = reg * Nx.reshape(alpha, {:auto, 1, 1})
      kernel = kernel + reg
      dual_coeff = Nx.LinAlg.solve(kernel, y) |> Nx.squeeze(axes: [-1])

      dual_coeff =
        if opts[:sample_weights_flag],
          do: dual_coeff * Nx.new_axis(sample_weights, 0),
          else: dual_coeff

      Nx.transpose(dual_coeff)
    end
  end

  defnp solve_cholesky(x, y, alpha) do
    num_features = Nx.axis_size(x, 1)
    num_targets = Nx.axis_size(y, 1)

    kernel = Nx.dot(x, [0], x, [0])

    xy = Nx.dot(x, [0], y, [0])

    one_alpha = Nx.all(alpha[[0]] == alpha)

    {kernel, xy}

    if one_alpha do
      kernel = kernel + Nx.eye(num_features) * alpha[[0]]
      Nx.LinAlg.solve(kernel, xy) |> Nx.transpose()
    else
      target = Nx.transpose(xy)

      # {_, _, _, coeff, _} =
      #   while {kernel, target, alpha, coeff = Nx.broadcast(0.0, {num_targets, num_features}),
      #          i = 0},
      #         i < num_targets do
      #     kernel = kernel + Nx.eye(num_features) * alpha[[i]]
      #     coeff_slice = Nx.LinAlg.solve(kernel, target[[i]]) |> Nx.new_axis(0)
      #     coeff = Nx.put_slice(coeff, [i, 0], coeff_slice)
      #     kernel = kernel - Nx.eye(num_features) * alpha[[i]]
      #     {kernel, target, alpha, coeff, i + 1}
      #   end

      target = Nx.new_axis(target, -1)

      kernel = Nx.new_axis(kernel, 0) |> Nx.broadcast({num_targets, num_features, num_features})

      reg =
        Nx.new_axis(Nx.eye(num_features), 0)
        |> Nx.broadcast({num_targets, num_features, num_features})

      reg = reg * Nx.reshape(alpha, {:auto, 1, 1})
      kernel = kernel + reg
      Nx.LinAlg.solve(kernel, target) |> Nx.squeeze(axes: [-1])
    end
  end

  defn solve_svd(x, y, alpha) do
    {u, s, vt} = Nx.LinAlg.svd(x)
    min_shape = Kernel.min(Nx.axis_size(u, -1), Nx.axis_size(vt, -1))
    u = Nx.slice_along_axis(u, 0, min_shape, axis: -1)
    vt = Nx.slice_along_axis(vt, 0, min_shape, axis: -2)
    s_size = Nx.size(s)
    alpha_size = Nx.size(alpha)
    idx = (s > 1.0e-15) |> Nx.new_axis(1) |> Nx.broadcast({s_size, alpha_size})
    s = Nx.new_axis(s, 1) |> Nx.broadcast({s_size, alpha_size})
    alpha = Nx.new_axis(alpha, 0) |> Nx.broadcast({s_size, alpha_size})
    d = Nx.broadcast(0.0, {s_size, alpha_size})
    d = Nx.select(idx, s / (s ** 2 + alpha), d)
    uty = Nx.dot(u, [0], y, [0])
    d_uty = d * uty
    Nx.dot(vt, [0], d_uty, [0]) |> Nx.transpose()
  end

  defnp set_intercept(coeff, x_offset, y_offset, fit_intercept?) do
    if fit_intercept? do
      y_offset - Nx.dot(coeff, x_offset)
    else
      0.0
    end
  end

  defn preprocess_data(x, y, sample_weights, opts) do
    if opts[:sample_weights_flag],
      do:
        {Nx.weighted_mean(x, sample_weights, axes: [0]),
         Nx.weighted_mean(y, sample_weights, axes: [0])},
      else: {Nx.mean(x, axes: [0]), Nx.mean(y, axes: [0])}
  end
end
