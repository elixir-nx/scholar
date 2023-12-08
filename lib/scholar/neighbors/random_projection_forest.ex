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
           keep: [:depth, :num_trees, :leaf_size],
           containers: [:indices, :data, :hyperplanes, :medians]}
  @enforce_keys [:depth, :num_trees, :leaf_size, :indices, :data, :hyperplanes, :medians]
  defstruct [:depth, :num_trees, :leaf_size, :indices, :data, :hyperplanes, :medians]

  def grow(tensor, num_trees, min_leaf_size) do
    key = Nx.Random.key(System.system_time())
    grow(tensor, num_trees, min_leaf_size, key)
  end

  def grow(tensor, num_trees, min_leaf_size, key) do
    tensor = to_float(tensor)
    {size, dim} = Nx.shape(tensor)
    {depth, leaf_size} = compute_depth_and_leaf_size(size, min_leaf_size, 0)
    num_nodes = 2 ** depth - 1

    {hyperplanes, _key} =
      Nx.Random.normal(key, type: Nx.type(tensor), shape: {num_trees, depth, dim})

    medians = Nx.broadcast(:nan, {num_trees, num_nodes})
    # TODO: Make acc unsigned int
    # TODO: Maybe rename acc to indices
    acc = Nx.broadcast(-1, {num_trees, size})
    root = 0
    start_index = 0
    indices = Nx.iota({size}) |> Nx.new_axis(0) |> Nx.broadcast({num_trees, size})

    {indices, medians} =
      recur([{root, start_index, indices}], [], acc, leaf_size, 0, tensor, hyperplanes, medians)

    %__MODULE__{
      depth: depth,
      num_trees: num_trees,
      leaf_size: leaf_size,
      indices: indices,
      data: tensor,
      hyperplanes: hyperplanes,
      medians: medians
    }
  end

  defp recur([], [], acc, _, _, _, _, medians) do
    {acc, medians}
  end

  defp recur([], next, acc, leaf_size, level, tensor, hyperplanes, medians) do
    recur(next, [], acc, leaf_size, level + 1, tensor, hyperplanes, medians)
  end

  defp recur(
         [{node, start_index, indices} | rest],
         next,
         acc,
         leaf_size,
         level,
         tensor,
         hyperplanes,
         medians
       ) do
    size = Nx.axis_size(indices, 1)

    if size > leaf_size do
      hyperplane = hyperplanes[[.., level]]

      {median, left_indices, right_indices} =
        Nx.Defn.jit_apply(&split(&1, &2, &3), [tensor, indices, hyperplane])

      medians = Nx.put_slice(medians, [0, node], Nx.new_axis(median, 1))
      child_start_index = start_index
      next = [{left_child(node), child_start_index, left_indices} | next]
      child_start_index = start_index + div(size, 2) + rem(size, 2)
      next = [{right_child(node), child_start_index, right_indices} | next]
      recur(rest, next, acc, leaf_size, level, tensor, hyperplanes, medians)
    else
      acc = Nx.put_slice(acc, [0, start_index], indices)
      recur(rest, next, acc, leaf_size, level, tensor, hyperplanes, medians)
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

  defp compute_depth_and_leaf_size(size, min_leaf_size, depth) do
    if div(size, 2) < min_leaf_size do
      {depth, size}
    else
      compute_depth_and_leaf_size(div(size, 2) + rem(size, 2), min_leaf_size, depth + 1)
    end
  end

  defn left_child(node) do
    2 * node + 1
  end

  defn right_child(node) do
    2 * node + 2
  end
end
