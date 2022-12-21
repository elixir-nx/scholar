defmodule Scholar.Linear.LinearRegression do
  @moduledoc """
  Ordinary least squares linear regression.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients]}
  defstruct [:coefficients]

  opts = [
    sample_weights: [
      type: {:list, {:custom, Scholar.Options, :positive_number, []}}
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a linear regression model for sample inputs `x` and
  sample targets `y`.
  """
  deftransform fit(x, y, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    opts =
      [
        sample_weights_flag: opts[:sample_weights] != nil
      ] ++
        opts

    {sample_weights, opts} = Keyword.pop(opts, :sample_weights, 1.0)
    sample_weights = Nx.tensor(sample_weights)

    fit_n(x, y, sample_weights, opts)
  end

  defnp fit_n(x, y, sample_weights, opts \\ []) do
    {x, y} = preprocess_data(x, y, sample_weights, opts)

    {x, y} =
      if opts[:sample_weights_flag] do
        rescale(x, y, sample_weights)
      else
        {x, y}
      end

    coeff = lstsq(x, y)
    %__MODULE__{coefficients: Nx.transpose(coeff)}
  end

  @doc """
  Makes predictions with the given model on inputs `x`.
  """
  defn predict(%__MODULE__{coefficients: coeff}, x) do
    Nx.dot(x, coeff)
  end

  # Implements sample weighting by rescaling inputs and
  # targets by sqrt(sample_weight).
  defnp rescale(x, y, sample_weights) do
    n = Nx.axis_size(x, 0)

    sample_weights =
      case Nx.shape(sample_weights) do
        {} ->
          Nx.broadcast(sample_weights, {n})

        _ ->
          sample_weights
      end

    scale = Nx.sqrt(sample_weights)
    scale = Nx.make_diagonal(scale)

    {Nx.dot(scale, x), Nx.dot(scale, y)}
  end

  # Implements ordinary least-squares by estimating the
  # solution A to the equation A.X = y.
  defnp lstsq(x, y) do
    y_rank = Nx.rank(y)
    y = if Nx.rank(y) == 1, do: Nx.new_axis(y, -1), else: y
    pinv = Nx.LinAlg.pinv(x)
    coeff = Nx.dot(pinv, y)
    if y_rank == 1, do: Nx.flatten(coeff), else: coeff
  end

  defnp preprocess_data(x, y, sample_weights, opts \\ []) do
    x_offset =
      if opts[:sample_weights_flag],
        do: Nx.weighted_mean(x, sample_weights, axis: 0),
        else: Nx.mean(x, axes: [0])

    x = x - x_offset

    y_offset =
      if opts[:sample_weights_flag],
        do: Nx.weighted_mean(y, sample_weights, axis: 0),
        else: Nx.mean(y, axes: [0])

    y = y - y_offset
    {x, y}
  end
end
