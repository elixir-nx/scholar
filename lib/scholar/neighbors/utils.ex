defmodule Scholar.Neighbors.Utils do
  @moduledoc false
  import Nx.Defn

  def metric(:cosine), do: {:ok, &Scholar.Metrics.Distance.cosine/2}

  def metric({:minkowski, p}) when p == :infinity or (is_number(p) and p > 0) do
    {:ok, &Scholar.Metrics.Distance.minkowski(&1, &2, p: p)}
  end

  def metric(metric) when is_function(metric, 2), do: {:ok, metric}

  def metric(metric) do
    {:error,
     "expected metric to be a 2-arity function, :cosine, tuple {:minkowski, p} where p is a positive number or :infinity, got: #{inspect(metric)}"}
  end

  def pairwise_metric(:cosine), do: {:ok, &Scholar.Metrics.Distance.pairwise_cosine/2}

  def pairwise_metric({:minkowski, p}) when p == :infinity or (is_number(p) and p > 0) do
    {:ok, &Scholar.Metrics.Distance.pairwise_minkowski(&1, &2, p: p)}
  end

  def pairwise_metric(:euclidean), do: {:ok, &Scholar.Metrics.Distance.pairwise_euclidean/2}

  def pairwise_metric(:squared_euclidean),
    do: {:ok, &Scholar.Metrics.Distance.pairwise_squared_euclidean/2}

  def pairwise_metric(:manhattan),
    do: {:ok, &Scholar.Metrics.Distance.pairwise_minkowski(&1, &2, p: 1)}

  def pairwise_metric(metric) when is_function(metric, 2), do: {:ok, metric}

  def pairwise_metric(metric) do
    {:error,
     "expected metric to be a 2-arity function, :cosine or tuple {:minkowski, p} where p is a positive number or :infinity, got: #{inspect(metric)}"}
  end

  defn brute_force_search_with_candidates(data, query, candidate_indices, opts) do
    k = opts[:num_neighbors]
    metric = opts[:metric]
    dim = Nx.axis_size(data, 1)
    {size, length} = Nx.shape(candidate_indices)

    x =
      query
      |> Nx.new_axis(1)
      |> Nx.broadcast({size, length, dim})
      |> Nx.vectorize([:query, :candidates])

    y = Nx.take(data, candidate_indices) |> Nx.vectorize([:query, :candidates])
    distances = metric.(x, y) |> Nx.devectorize() |> Nx.rename(nil)

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

    target_indices =
      Nx.concatenate([samples, Nx.new_axis(indices, 2)], axis: 2)
      |> Nx.reshape({size * length, 2})

    updates = Nx.iota({size, length}, axis: 1) |> Nx.reshape({size * length})
    Nx.indexed_add(target, target_indices, updates)
  end

  defn check_weights(weights) do
    zero_mask = weights == 0
    zero_rows = zero_mask |> Nx.any(axes: [1], keep_axes: true) |> Nx.broadcast(weights)
    weights = Nx.select(zero_mask, 1, weights)
    weights_inv = 1 / weights
    Nx.select(zero_rows, zero_mask, weights_inv)
  end
end
