defmodule Scholar.Linear.LinearRegression do
  @moduledoc """
  Ordinary least squares linear regression.
  """
  import Nx.Defn
  alias __MODULE__, as: LinearRegression

  @derive {Nx.Container, containers: [:coefficients]}
  defstruct [:coefficients]

  @doc """
  Fits a linear regression model for sample inputs `x` and
  sample targets `y`.
  """
  defn fit(x, y, opts \\ []) do
    opts = keyword!(opts, sample_weight: nil)

    {x, y} =
      transform({x, y, opts[:sample_weight]}, fn
        {x, y, nil} -> {x, y}
        {x, y, sample_weight} -> rescale(x, y, sample_weight)
      end)

    coef = lstsq(x, y)

    %LinearRegression{coefficients: coef}
  end

  @doc """
  Makes predictions with the given model on inputs `x`.
  """
  defn predict(%LinearRegression{coefficients: coeff}, x) do
    Nx.dot(x, coeff)
  end

  # Implements sample weighting by rescaling inputs and
  # targets by sqrt(sample_weight).
  defnp rescale(x, y, sample_weight) do
    n = Nx.axis_size(x, 0)

    scale =
      sample_weight
      |> Nx.broadcast({n})
      |> Nx.sqrt()
      |> Nx.make_diagonal()

    {Nx.dot(x, scale), Nx.dot(y, scale)}
  end

  # Implements ordinary least-squares by estimating the
  # solution A to the equation A.X = b.
  defnp lstsq(x, y) do
    {u, s, vt} = Nx.LinAlg.svd(x)

    mask = Nx.not_equal(s, 0)
    safe_s = Nx.select(mask, s, 1)
    s_inv = Nx.divide(1, safe_s)

    vt
    |> Nx.transpose()
    |> Nx.multiply(s_inv)
    |> Nx.dot(Nx.transpose(u))
    |> Nx.dot(y)
  end
end
