defmodule Scholar.Cluster.Utils do
  import Nx.Defn
  import Scholar.Shared

  @spec squared_euclidean(Nx.t(), Nx.t()) :: Nx.t()
  defn squared_euclidean(x, y, opts \\ []) do
    assert_same_shape!(x, y)

    opts =
      keyword!(
        opts,
        axes: 0
      )

    x
    |> Nx.subtract(y)
    |> Nx.power(2)
    |> Nx.sum(axes: [opts[:axes]])
    |> Nx.as_type({:f, 32})
  end
end
