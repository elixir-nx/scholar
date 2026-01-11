defmodule Scholar.Covariance.Utils do
  @moduledoc false
  import Nx.Defn

  defn center(x, assume_centered? \\ false) do
    x =
      case Nx.shape(x) do
        {_} -> Nx.new_axis(x, 1)
        _ -> x
      end

    location =
      if assume_centered? do
        0
      else
        Nx.mean(x, axes: [0])
      end

    {x - location, location}
  end

  defn empirical_covariance(x) do
    n = Nx.axis_size(x, 0)

    covariance = Nx.dot(x, [0], x, [0]) / n

    case Nx.shape(covariance) do
      {} -> Nx.reshape(covariance, {1, 1})
      _ -> covariance
    end
  end

  defn trace(x) do
    x
    |> Nx.take_diagonal()
    |> Nx.sum()
  end
end
