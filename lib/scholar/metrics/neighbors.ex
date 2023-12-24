defmodule Scholar.Metrics.Neighbors do
  import Nx.Defn

  deftransform recall(graph_true, graph_pred) do
    # TODO: Validate graph_true and graph_pred
    recall_n(graph_true, graph_pred)
  end

  defn recall_n(graph_true, graph_pred) do
    {n, k} = Nx.shape(graph_true)
    concatenated = Nx.concatenate([graph_true, graph_pred], axis: 1) |> Nx.sort(axis: 1)
    duplicate_mask = concatenated[[.., 0..(2 * k - 2)]] == concatenated[[.., 1..(2 * k - 1)]]
    duplicate_mask |> Nx.sum() |> Nx.divide(n * k)
  end
end
