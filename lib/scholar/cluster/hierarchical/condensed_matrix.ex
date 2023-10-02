defmodule Scholar.Cluster.Hierarchical.CondensedMatrix do
  import Nx.Defn

  # The elements of `d` below the diagonal flattened into a 1D tensor.
  def condense_pairwise(%Nx.Tensor{} = d) do
    {n, n} = Nx.shape(d)
    Nx.gather(d, pairwise_indices(n))
  end

  def pairwise_indices(n), do: Nx.tensor(for(i <- 1..(n - 1), j <- 0..(i - 1), do: [i, j]))

  defn rc_to_i(%Nx.Tensor{} = rc), do: tri(rc[[.., 0]] - 1) + rc[[.., 1]]

  defn i_to_rc(%Nx.Tensor{} = i) do
    t = tri_inv(i)
    Nx.stack([i - tri(t), t + 1])
  end

  # nth triangle number, i.e. (n choose 2)
  defn tri(n), do: Nx.quotient(n * (n + 1), 2)

  # largest n such that tri(n) <= t
  defn tri_inv(t), do: Nx.as_type((Nx.sqrt(8 * t + 1) - 1) / 2, :u32)
end
