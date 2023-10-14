defmodule Scholar.Cluster.Hierarchical.CondensedMatrix do
  import Nx.Defn

  # The elements of `d` below the diagonal flattened into a 1D tensor.
  defn condense_pairwise(%Nx.Tensor{} = d) do
    {n, _} = Nx.shape(d)
    Nx.gather(d, pairwise_indices(n))
  end

  # `tri(n) * 2` tensor of row/col indices for the below-diagonal.
  deftransform pairwise_indices(n) do
    {div(n * (n - 1), 2)}
    |> Nx.iota()
    |> i_to_rc()
  end

  defn rc_to_i(%Nx.Tensor{} = rc), do: tri(rc[[.., 0]] - 1) + rc[[.., 1]]

  defn i_to_rc(%Nx.Tensor{} = i) do
    n = tri_inv(i)
    Nx.stack([n + 1, i - tri(n)], axis: 1)
  end

  # nth triangle number, i.e. (n choose 2)
  defn tri(n), do: Nx.quotient(n * (n + 1), 2)

  # largest n such that tri(n) <= t
  defn tri_inv(t), do: Nx.as_type((Nx.sqrt(8 * t + 1) - 1) / 2, :u32)
end
