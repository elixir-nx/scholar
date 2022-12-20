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

  @eps 1.1920929e-07

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
    {m, n} = Nx.shape(x)
    rcond = @eps * Nx.max(m, n)

    {u, s, vt} = Nx.LinAlg.svd(x)

    y_rank = Nx.rank(y)
    y = if Nx.rank(y) == 1, do: Nx.new_axis(y, -1), else: y

    {u_rows, _u_cols} = Nx.shape(u)
    {vt_rows, _vt_cols} = Nx.shape(vt)
    min = Kernel.min(u_rows, vt_rows)
    vt = vt[[0..(min - 1), 0..-1//1]]
    u = u[[0..-1//1, 0..(min - 1)]]
    mask = s > rcond * Nx.reduce_max(s)
    safe_s = Nx.select(mask, s, 1)

    s_inv = Nx.new_axis(Nx.select(mask, 1 / safe_s, 0), -1)

    utb = Nx.dot(Nx.LinAlg.adjoint(u), y)
    coeffs = Nx.dot(Nx.LinAlg.adjoint(vt), s_inv * utb)
    if y_rank == 1, do: Nx.flatten(coeffs), else: coeffs
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
