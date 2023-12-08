defmodule Scholar.Neighbors.RandomProjectionForest do
  @moduledoc """
  Random Projection Forest.

  ...

  * [1] - Random projection trees and low dimensional manifolds
  * [2] - Fast Nearest Neighbor Search through Sparse Random Projections and Voting
  """

  import Nx.Defn
  import Scholar.Shared
  require Nx

  @derive {Nx.Container,
           keep: [:height, :num_trees, :leaf_size],
           containers: [:indices, :data, :hyperplanes, :medians]}
  @enforce_keys [:height, :num_trees, :leaf_size, :indices, :data, :hyperplanes, :medians]
  defstruct [:height, :num_trees, :leaf_size, :indices, :data, :hyperplanes, :medians]

  def grow(tensor, num_trees, min_leaf_size) do
    key = Nx.Random.key(System.system_time())
    grow(tensor, num_trees, min_leaf_size, key)
  end

  def grow(tensor, num_trees, min_leaf_size, key) do
    tensor = to_float(tensor)
    {size, dim} = Nx.shape(tensor)
    {height, leaf_size} = compute_height_and_leaf_size(size, min_leaf_size, 0)
    num_nodes = 2 ** height - 1

    {hyperplanes, _key} =
      Nx.Random.normal(key, type: Nx.type(tensor), shape: {num_trees, num_nodes, dim})

    medians = Nx.broadcast(:nan, {num_trees, num_nodes})
    acc = Nx.broadcast(-1, {num_trees, size})
    root = 0
    start_index = 0
    indices = Nx.iota({size}) |> Nx.new_axis(0) |> Nx.broadcast({num_trees, size})

    {indices, medians} =
      recur([{root, start_index, indices}], [], acc, leaf_size, tensor, hyperplanes, medians)

    %__MODULE__{
      height: height,
      num_trees: num_trees,
      leaf_size: leaf_size,
      indices: indices,
      data: tensor,
      hyperplanes: hyperplanes,
      medians: medians
    }
  end

  defp recur([], [], acc, _, _, _, medians) do
    {acc, medians}
  end

  defp recur([], next, acc, leaf_size, tensor, hyperplanes, medians) do
    recur(next, [], acc, leaf_size, tensor, hyperplanes, medians)
  end

  defp recur(
         [{node, start_index, indices} | rest],
         next,
         acc,
         leaf_size,
         tensor,
         hyperplanes,
         medians
       ) do
    size = Nx.axis_size(indices, 1)

    if size > leaf_size do
      hyperplane = hyperplanes[[.., node]]

      {median, left_indices, right_indices} =
        Nx.Defn.jit_apply(&split(&1, &2, &3), [tensor, indices, hyperplane])

      medians = Nx.put_slice(medians, [0, node], Nx.new_axis(median, 1))
      child_start_index = start_index
      next = [{left_child(node), child_start_index, left_indices} | next]
      child_start_index = start_index + div(size, 2) + rem(size, 2)
      next = [{right_child(node), child_start_index, right_indices} | next]
      recur(rest, next, acc, leaf_size, tensor, hyperplanes, medians)
    else
      acc = Nx.put_slice(acc, [0, start_index], indices)
      recur(rest, next, acc, leaf_size, tensor, hyperplanes, medians)
    end
  end

  defnp split(tensor, indices, hyperplane) do
    size = Nx.axis_size(indices, 1)
    slice = Nx.take(tensor, indices)
    proj = Nx.dot(slice, [2], [0], hyperplane, [1], [0])
    sorted_indices = Nx.argsort(proj, axis: 1)
    proj = Nx.take_along_axis(proj, sorted_indices, axis: 1)

    median =
      if rem(size, 2) == 1 do
        proj[[.., div(size, 2)]]
      else
        # TODO: Maybe rewrite this!
        Nx.mean(Nx.slice_along_axis(proj, div(size, 2) - 1, 2, axis: 1), axes: [1])
      end

    indices = Nx.take_along_axis(indices, sorted_indices, axis: 1)
    left_size = div(size, 2) + rem(size, 2)
    right_size = div(size, 2)
    left_indices = Nx.slice_along_axis(indices, 0, left_size, axis: 1)
    right_indices = Nx.slice_along_axis(indices, left_size, right_size, axis: 1)

    {median, left_indices, right_indices}
  end

  defp compute_height_and_leaf_size(size, min_leaf_size, height) do
    if div(size, 2) < min_leaf_size do
      {height, size}
    else
      compute_height_and_leaf_size(div(size, 2) + rem(size, 2), min_leaf_size, height + 1)
    end
  end

  defn left_child(node) do
    2 * node + 1
  end

  defn right_child(node) do
    2 * node + 2
  end
end
