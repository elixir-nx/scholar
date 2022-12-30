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
  Fits a linear regression model for sample inputs `a` and
  sample targets `b`.
  """
  deftransform fit(a, b, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    opts =
      [
        sample_weights_flag: opts[:sample_weights] != nil
      ] ++
        opts

    {sample_weights, opts} = Keyword.pop(opts, :sample_weights, 1.0)
    sample_weights = Nx.tensor(sample_weights)

    fit_n(a, b, sample_weights, opts)
  end

  defnp fit_n(a, b, sample_weights, opts \\ []) do
    {a, b} = preprocess_data(a, b, sample_weights, opts)

    {a, b} =
      if opts[:sample_weights_flag] do
        rescale(a, b, sample_weights)
      else
        {a, b}
      end

    coeff = lstsq(a, b)
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

    scale = Nx.sqrt(sample_weights) |> Nx.make_diagonal()

    {Nx.dot(scale, x), Nx.dot(scale, y)}
  end

  # Implements ordinary least-squares by estimating the
  # solution A to the equation A.X = b.
  defnp lstsq(a, b) do
    pinv = Nx.LinAlg.pinv(a)
    Nx.dot(pinv, b)
  end

  defnp preprocess_data(x, y, sample_weights, opts \\ []) do
    {x_offset, y_offset} =
      if opts[:sample_weights_flag],
        do:
          {Nx.weighted_mean(x, sample_weights, axes: [0]),
           Nx.weighted_mean(y, sample_weights, axes: [0])},
        else: {Nx.mean(x, axes: [0]), Nx.mean(y, axes: [0])}

    {x - x_offset, y - y_offset}
  end
end
