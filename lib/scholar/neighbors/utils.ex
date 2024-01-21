defmodule Scholar.Neighbors.Utils do
  @moduledoc false
  import Nx.Defn
  require Nx

  defn find_neighbors(query, data, candidate_indices, opts) do
    k = opts[:num_neighbors]
    {size, length} = Nx.shape(candidate_indices)

    distances =
      query
      |> Nx.new_axis(1)
      |> Nx.subtract(Nx.take(data, candidate_indices))
      |> Nx.pow(2)
      |> Nx.sum(axes: [2])

    distances =
      if length > 1 do
        sorted_indices = Nx.argsort(candidate_indices, axis: 1, stable: true)
        inverse = inverse_permutation(sorted_indices)
        sorted = Nx.take_along_axis(candidate_indices, sorted_indices, axis: 1)

        duplicate_mask =
          Nx.concatenate(
            [
              Nx.broadcast(0, {size, 1}),
              Nx.equal(sorted[[.., 0..-2//1]], sorted[[.., 1..-1//1]])
            ],
            axis: 1
          )
          |> Nx.take_along_axis(inverse, axis: 1)

        Nx.select(duplicate_mask, :infinity, distances)
      else
        distances
      end

    indices = Nx.argsort(distances, axis: 1) |> Nx.slice_along_axis(0, k, axis: 1)

    neighbor_indices =
      Nx.take(
        Nx.vectorize(candidate_indices, :samples),
        Nx.vectorize(indices, :samples)
      )
      |> Nx.devectorize()
      |> Nx.rename(nil)

    neighbor_distances = Nx.take_along_axis(distances, indices, axis: 1)

    {neighbor_indices, neighbor_distances}
  end

  defnp inverse_permutation(indices) do
    {size, length} = Nx.shape(indices)
    target = Nx.broadcast(Nx.u32(0), {size, length})
    samples = Nx.iota({size, length, 1}, axis: 0)

    indices =
      Nx.concatenate([samples, Nx.new_axis(indices, 2)], axis: 2)
      |> Nx.reshape({size * length, 2})

    updates = Nx.iota({size, length}, axis: 1) |> Nx.reshape({size * length})
    Nx.indexed_add(target, indices, updates)
  end
end
