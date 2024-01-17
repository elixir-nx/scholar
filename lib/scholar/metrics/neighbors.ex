defmodule Scholar.Metrics.Neighbors do
  import Nx.Defn

  deftransform recall(graph_true, graph_pred) do
    if Nx.rank(graph_true) != 2 do
      raise ArgumentError,
            """
            expected true neighbors to have shape {num_samples, num_neighbors}, \
            got tensor with shape: #{inspect(Nx.shape(graph_true))}\
            """
    end

    if Nx.rank(graph_pred) != 2 do
      raise ArgumentError,
            """
            expected predicted neighbors to have shape {num_samples, num_neighbors}, \
            got tensor with shape: #{inspect(Nx.shape(graph_pred))}\
            """
    end

    if Nx.shape(graph_true) != Nx.shape(graph_pred) do
      raise ArgumentError,
            """
            expected true and predicted neighbors to have the same shape, \
            got #{inspect(Nx.shape(graph_true))} and #{inspect(Nx.shape(graph_pred))}\
            """
    end

    recall_n(graph_true, graph_pred)
  end

  defn recall_n(graph_true, graph_pred) do
    {n, k} = Nx.shape(graph_true)
    concatenated = Nx.concatenate([graph_true, graph_pred], axis: 1) |> Nx.sort(axis: 1)
    duplicate_mask = concatenated[[.., 0..(2 * k - 2)]] == concatenated[[.., 1..(2 * k - 1)]]
    duplicate_mask |> Nx.sum() |> Nx.divide(n * k)
  end
end
