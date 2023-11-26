defmodule Scholar.Linear.LinearRegression do
  @moduledoc """
  Ordinary least squares linear regression.

  Time complexity of linear regression is $O((K^2) * (K+N))$ where $N$ is the number of samples
  and $K$ is the number of features.
  """
  require Nx
  import Nx.Defn
  import Scholar.Shared

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
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a linear regression model for sample inputs `x` and
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
      iex> model = Scholar.Linear.LinearRegression.fit(x, y)
      iex> model.coefficients
      #Nx.Tensor<
        f32[2]
        [-0.49724727869033813, -0.7010392546653748]
      >
      iex> model.intercept
      #Nx.Tensor<
        f32
        5.896470069885254
      >
  """
  deftransform fit(x, y, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    opts =
      [
        sample_weights_flag: opts[:sample_weights] != nil
      ] ++
        opts

    {sample_weights, opts} = Keyword.pop(opts, :sample_weights, 1.0)
    x_type = to_float_type(x)

    sample_weights =
      if Nx.is_tensor(sample_weights),
        do: Nx.as_type(sample_weights, x_type),
        else: Nx.tensor(sample_weights, type: x_type)

    fit_n(x, y, sample_weights, opts)
  end

  defnp fit_n(a, b, sample_weights, opts) do
    a = to_float(a)
    b = to_float(b)

    {a_offset, b_offset} =
      if opts[:fit_intercept?] do
        preprocess_data(a, b, sample_weights, opts)
      else
        a_offset_shape = Nx.axis_size(a, 1)
        b_reshaped = if Nx.rank(b) > 1, do: b, else: Nx.reshape(b, {:auto, 1})
        b_offset_shape = Nx.axis_size(b_reshaped, 1)

        {Nx.broadcast(Nx.tensor(0.0, type: Nx.type(a)), {a_offset_shape}),
         Nx.broadcast(Nx.tensor(0.0, type: Nx.type(b)), {b_offset_shape})}
      end

    {a, b} = {a - a_offset, b - b_offset}

    {a, b} =
      if opts[:sample_weights_flag] do
        rescale(a, b, sample_weights)
      else
        {a, b}
      end

    {coeff, intercept} = lstsq(a, b, a_offset, b_offset, opts[:fit_intercept?])
    %__MODULE__{coefficients: coeff, intercept: intercept}
  end

  @doc """
  Makes predictions with the given `model` on input `x`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> model = Scholar.Linear.LinearRegression.fit(x, y)
      iex> Scholar.Linear.LinearRegression.predict(model, Nx.tensor([[2.0, 1.0]]))
      Nx.tensor(
        [4.200936317443848]
      )
  """
  defn predict(%__MODULE__{coefficients: coeff, intercept: intercept} = _model, x) do
    Nx.dot(x, coeff) + intercept
  end

  # Implements sample weighting by rescaling inputs and
  # targets by sqrt(sample_weight).
  defnp rescale(x, y, sample_weights) do
    case Nx.shape(sample_weights) do
      {} = scalar ->
        scalar = Nx.sqrt(scalar)
        {scalar * x, scalar * y}

      _ ->
        scale = sample_weights |> Nx.sqrt() |> Nx.make_diagonal()
        {Nx.dot(scale, x), Nx.dot(scale, y)}
    end
  end

  # Implements ordinary least-squares by estimating the
  # solution A to the equation A.X = b.
  defnp lstsq(a, b, a_offset, b_offset, fit_intercept?) do
    pinv = Nx.LinAlg.pinv(a)
    coeff = Nx.dot(b, [0], pinv, [1])
    intercept = set_intercept(coeff, a_offset, b_offset, fit_intercept?)
    {coeff, intercept}
  end

  defnp set_intercept(coeff, x_offset, y_offset, fit_intercept?) do
    if fit_intercept? do
      y_offset - Nx.dot(coeff, x_offset)
    else
      Nx.tensor(0.0, type: Nx.type(coeff))
    end
  end

  defnp preprocess_data(x, y, sample_weights, opts) do
    if opts[:sample_weights_flag],
      do:
        {Nx.weighted_mean(x, sample_weights, axes: [0]),
         Nx.weighted_mean(y, sample_weights, axes: [0])},
      else: {Nx.mean(x, axes: [0]), Nx.mean(y, axes: [0])}
  end
end
