defmodule Scholar.Metrics.Neighbors do
  @moduledoc """
  Metrics for evaluating the results of approximate k-nearest neighbor search algorithms.
  """

  import Nx.Defn

  @doc """
  Computes the recall of predicted k-nearest neighbors given the true k-nearest neighbors.
  Recall is defined as the average fraction of nearest neighbors the algorithm predicted correctly.

  ## Examples

      iex> neighbors_true = Nx.tensor([[0, 1], [1, 2], [2, 1]])
      iex> Scholar.Metrics.Neighbors.recall(neighbors_true, neighbors_true)
      #Nx.Tensor<
        f32
        1.0
      >

      iex> neighbors_true = Nx.tensor([[0, 1], [1, 2], [2, 1]])
      iex> neighbors_pred = Nx.tensor([[0, 1], [1, 0], [2, 0]])
      iex> Scholar.Metrics.Neighbors.recall(neighbors_true, neighbors_pred)
      #Nx.Tensor<
        f32
        0.6666666865348816
      >
  """
  defn recall(neighbors_true, neighbors_pred) do
    if Nx.rank(neighbors_true) != 2 do
      raise ArgumentError,
            """
            expected true neighbors to have shape {num_samples, num_neighbors}, \
            got tensor with shape: #{inspect(Nx.shape(neighbors_true))}\
            """
    end

    if Nx.rank(neighbors_pred) != 2 do
      raise ArgumentError,
            """
            expected predicted neighbors to have shape {num_samples, num_neighbors}, \
            got tensor with shape: #{inspect(Nx.shape(neighbors_pred))}\
            """
    end

    if Nx.axis_size(neighbors_true, 0) != Nx.axis_size(neighbors_pred, 0) do
      raise ArgumentError,
            """
            expected true and predicted neighbors to have the same axis 0 size, \
            got #{inspect(Nx.axis_size(neighbors_true, 0))} and #{inspect(Nx.axis_size(neighbors_pred, 0))}\
            """
    end

    if Nx.axis_size(neighbors_true, 1) != Nx.axis_size(neighbors_pred, 1) do
      raise ArgumentError,
            """
            expected true and predicted neighbors to have the same axis 1 size, \
            got #{inspect(Nx.axis_size(neighbors_true, 1))} and #{inspect(Nx.axis_size(neighbors_pred, 1))}\
            """
    end

    {n, k} = Nx.shape(neighbors_true)
    concatenated = Nx.concatenate([neighbors_true, neighbors_pred], axis: 1) |> Nx.sort(axis: 1)
    duplicate_mask = concatenated[[.., 0..(2 * k - 2)]] == concatenated[[.., 1..(2 * k - 1)]]
    duplicate_mask |> Nx.sum() |> Nx.divide(n * k)
  end
end
