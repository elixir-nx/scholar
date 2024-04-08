defmodule Scholar.Neighbors.Utils do
  @moduledoc false
  import Nx.Defn
  require Nx
  import Scholar.Shared

  defn linear_search(data, query, opts) do
    k = opts[:num_neighbors]
    metric = opts[:metric]
    batch_size = opts[:batch_size]

    type = Nx.Type.merge(to_float_type(data), to_float_type(query))

    distance_fn =
      case metric do
        {:minkowski, p} ->
          fn data, query ->
            query
            |> Nx.new_axis(1)
            |> Nx.subtract(data)
            |> Nx.pow(p)
            |> Nx.sum(axes: [2])
          end

        {:cosine} ->
          &Scholar.Metrics.Distance.pairwise_cosine/2
      end

    {query_size, dim} = Nx.shape(query)
    num_batches = div(query_size, batch_size)
    leftover_size = rem(query_size, batch_size)

    batches =
      query
      |> Nx.slice_along_axis(0, num_batches * batch_size, axis: 0)
      |> Nx.reshape({num_batches, batch_size, dim})

    {neighbor_indices, neighbor_distances, _} =
      while {
              neighbor_indices = Nx.broadcast(Nx.u64(0), {num_batches, batch_size, k}),
              neighbor_distances =
                Nx.broadcast(Nx.as_type(:nan, type), {num_batches, batch_size, k}),
              {
                data,
                batches,
                i = 0
              }
            },
            i < num_batches do
        batch = batches[i]
        distances = distance_fn.(data, batch)

        indices = Nx.argsort(distances, axis: 1, type: :u64) |> Nx.slice_along_axis(0, k, axis: 1)
        distances = Nx.take_along_axis(distances, indices, axis: 1)

        neighbor_indices = Nx.put_slice(neighbor_indices, [i, 0, 0], Nx.new_axis(indices, 0))

        neighbor_distances =
          Nx.put_slice(neighbor_distances, [i, 0, 0], Nx.new_axis(distances, 0))

        {neighbor_indices, neighbor_distances, {data, batches, i + 1}}
      end

    neighbor_indices = Nx.reshape(neighbor_indices, {num_batches * batch_size, k})
    neighbor_distances = Nx.reshape(neighbor_distances, {num_batches * batch_size, k})

    {neighbor_indices, neighbor_distances} =
      if leftover_size > 0 do
        leftover = Nx.slice_along_axis(query, query_size - leftover_size, leftover_size, axis: 0)
        distances = distance_fn.(data, leftover)
        indices = Nx.argsort(distances, axis: 1, type: :u64) |> Nx.slice_along_axis(0, k, axis: 1)
        distances = Nx.take_along_axis(distances, indices, axis: 1)
        neighbor_indices = Nx.concatenate([neighbor_indices, indices])
        neighbor_distances = Nx.concatenate([neighbor_distances, distances])
        {neighbor_indices, neighbor_distances}
      else
        {neighbor_indices, neighbor_distances}
      end

    {neighbor_indices, neighbor_distances}
  end

  defn find_neighbors(query, data, candidate_indices, opts) do
    k = opts[:num_neighbors]
    {size, length} = Nx.shape(candidate_indices)

    # TODO: Add more distances!
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
