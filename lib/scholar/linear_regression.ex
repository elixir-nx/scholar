defmodule Scholar.LinearRegression do

  defstruct :coefficients

  @doc """
  Fits a linear regression model to the given dataset.
  """
  defn fit(x, y) do
    {coeff, intercept} = lstsq(x, y)
  end

  defnp lstsq(x, y) do
    {u, s, vt} = Nx.LinAlg.svd(x)

    uty =
      u
      |> Nx.transpose()
      |> Nx.dot(y)

    s_inv = Nx.new_axis(1 / s, -1)

    x =
      vt
      |> Nx.transpose()
      |> Nx.dot(Nx.multiply(s_inv, uty))

    Nx.transpose(x)
  end

  defimpl Scholar do
    import Nx.Defn

    def predict(%LinearRegression{coefficients: coef}, x) do
      decision_function(coeff, intercept, x)
    end

    defnp decision_function(coeff, x) do
      x
      |> Nx.dot(coeff)
    end
  end
end