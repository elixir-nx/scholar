defmodule Scholar.Metrics.Neighbors do
  import Nx.Defn

  deftransform recall(neighbors_true, neighbors_pred) do
    if Nx.rank(neighbors_true) != 2 do
      raise ArgumentError,
            """
            expected true neighbors to have shape {num_samples, num_neighbors}, \
            got tensor with shape: #{inspect(Nx.shape(neighbors_true))}\
            """
    end

<<<<<<< HEAD
    if Nx.rank(graph_pred) != 2 do
=======
    if Nx.rank(neighbors_pred) != 2 do
>>>>>>> 9d778a1 (Update.)
      raise ArgumentError,
            """
            expected predicted neighbors to have shape {num_samples, num_neighbors}, \
            got tensor with shape: #{inspect(Nx.shape(neighbors_pred))}\
            """
    end

<<<<<<< HEAD
    if Nx.shape(graph_true) != Nx.shape(graph_pred) do
=======
    if Nx.shape(neighbors_true) != Nx.shape(neighbors_pred) do
>>>>>>> 9d778a1 (Update.)
      raise ArgumentError,
            """
            expected true and predicted neighbors to have the same shape, \
            got #{inspect(Nx.shape(neighbors_true))} and #{inspect(Nx.shape(neighbors_pred))}\
            """
    end

<<<<<<< HEAD
    recall_n(graph_true, graph_pred)
=======
    recall_n(neighbors_true, neighbors_pred)
>>>>>>> 9d778a1 (Update.)
  end

  defn recall_n(neighbors_true, neighbors_pred) do
    {n, k} = Nx.shape(neighbors_true)
    concatenated = Nx.concatenate([neighbors_true, neighbors_pred], axis: 1) |> Nx.sort(axis: 1)
    duplicate_mask = concatenated[[.., 0..(2 * k - 2)]] == concatenated[[.., 1..(2 * k - 1)]]
    duplicate_mask |> Nx.sum() |> Nx.divide(n * k)
  end
end
