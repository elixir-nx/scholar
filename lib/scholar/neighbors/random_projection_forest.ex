defmodule Scholar.Neighbors.RandomProjectionForest do
  @moduledoc """
  Random Projection Forest.

  Each tree in a forest is constructed using a divide and conquer approach.
  We start with the entire dataset and at every node we project the data onto a random
  hyperplane and split it in the following way: the points with the projection smaller
  than or equal to the median are put into the left subtree and the points with projection
  greater than the median are put into the right subtree. We then proceed
  recursively with the left and right subtree.
  In this implementation the trees are complete, i.e. there are 2^l nodes at level l.
  The leaves of the trees are arranged as blocks in the field `indices`. We use the same
  hyperplane for all nodes on the same level as in [2].

  * [1] - Random projection trees and low dimensional manifolds
  * [2] - Fast Nearest Neighbor Search through Sparse Random Projections and Voting
  """

  import Nx.Defn
  import Scholar.Shared
  require Nx

  @derive {Nx.Container,
           keep: [:depth, :leaf_size, :num_trees],
           containers: [:indices, :data, :hyperplanes, :medians]}
  @enforce_keys [:depth, :leaf_size, :num_trees, :indices, :data, :hyperplanes, :medians]
  defstruct [:depth, :leaf_size, :num_trees, :indices, :data, :hyperplanes, :medians]

  opts = [
    num_trees: [
      required: true,
      type: :pos_integer,
      doc: "The number of trees in the forest."
    ],
    min_leaf_size: [
      required: true,
      type: :pos_integer,
      doc: "The minumum number of points in the leaf."
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Used for random number generation in hyperplane initialization.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Grows a random projection forest.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Examples

      iex> key = Nx.Random.key(12)
      iex> tensor = Nx.iota({5, 2})
      iex> Scholar.Neighbors.RandomProjectionForest.fit_unbounded(tensor, num_trees: 3, min_leaf_size: 2, key: key).indices
      #Nx.Tensor<
        u32[3][5]
        [
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [4, 3, 2, 1, 0]
        ]
      >
  """
  def fit_unbounded(tensor, opts) do
    if Nx.rank(tensor) != 2 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(tensor))}\
            """
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)
    min_leaf_size = opts[:min_leaf_size]
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)

    # if min_leaf_size == 1 do
    #   raise ArgumentError,
    #         """
    #         expected min_leaf_size to be at least 2, got 1
    #         """
    # end

    {size, dim} = Nx.shape(tensor)
    {depth, leaf_size} = compute_depth_and_leaf_size(size, min_leaf_size, 0)

    if depth == 0 do
      raise ArgumentError,
            """
            expected num_samples to be at least twice \
            min_leaf_size = #{inspect(min_leaf_size)}, got #{inspect(size)}
            """
    end

    num_trees = opts[:num_trees]
    num_nodes = 2 ** depth - 1

    {hyperplanes, _key} =
      Nx.Random.normal(key, type: to_float_type(tensor), shape: {num_trees, depth, dim})

    medians = Nx.broadcast(:nan, {num_trees, num_nodes})
    # TODO: Maybe rename acc to indices
    acc = Nx.broadcast(Nx.u32(0), {num_trees, size})
    root = 0
    start_index = 0
    indices = Nx.iota({1, size}, type: :u32) |> Nx.broadcast({num_trees, size})

    {indices, medians} =
      recur([{root, start_index, indices}], [], acc, depth, 0, tensor, hyperplanes, medians)

    %__MODULE__{
      depth: depth,
      leaf_size: leaf_size,
      num_trees: num_trees,
      indices: indices,
      data: tensor,
      hyperplanes: hyperplanes,
      medians: medians
    }
  end

  defp recur([], [], acc, _, _, _, _, medians) do
    {acc, medians}
  end

  defp recur([], next, acc, depth, level, tensor, hyperplanes, medians) do
    recur(next, [], acc, depth, level + 1, tensor, hyperplanes, medians)
  end

  defp recur(
         [{node, start_index, indices} | rest],
         next,
         acc,
         depth,
         level,
         tensor,
         hyperplanes,
         medians
       ) do
    size = Nx.axis_size(indices, 1)

    if level < depth do
      hyperplane = hyperplanes[[.., level]]

      {median, left_indices, right_indices} =
        Nx.Defn.jit_apply(&split(&1, &2, &3), [tensor, indices, hyperplane])

      medians = Nx.put_slice(medians, [0, node], Nx.new_axis(median, 1))
      child_start_index = start_index
      next = [{left_child(node), child_start_index, left_indices} | next]
      child_start_index = start_index + div(size, 2) + rem(size, 2)
      next = [{right_child(node), child_start_index, right_indices} | next]
      recur(rest, next, acc, depth, level, tensor, hyperplanes, medians)
    else
      acc = Nx.put_slice(acc, [0, start_index], indices)
      recur(rest, next, acc, depth, level, tensor, hyperplanes, medians)
    end
  end

  defnp split(tensor, indices, hyperplane) do
    size = Nx.axis_size(indices, 1)
    slice = Nx.take(tensor, indices)
    proj = Nx.dot(slice, [2], [0], hyperplane, [1], [0])
    sorted_indices = Nx.argsort(proj, axis: 1, type: :u32)
    proj = Nx.take_along_axis(proj, sorted_indices, axis: 1)

    right_size = div(size, 2)
    left_size = right_size + rem(size, 2)

    median =
      if left_size > right_size do
        proj[[.., right_size]]
      else
        two_mid_elems = Nx.slice_along_axis(proj, right_size - 1, 2, axis: 1)
        Nx.mean(two_mid_elems, axes: [1])
      end

    indices = Nx.take_along_axis(indices, sorted_indices, axis: 1)
    left_indices = Nx.slice_along_axis(indices, 0, left_size, axis: 1)
    right_indices = Nx.slice_along_axis(indices, left_size, right_size, axis: 1)

    {median, left_indices, right_indices}
  end

  defp compute_depth_and_leaf_size(size, min_leaf_size, depth) do
    right_size = div(size, 2)
    left_size = right_size + rem(size, 2)

    cond do
      right_size < min_leaf_size -> {depth, size}
      right_size == min_leaf_size -> {depth + 1, left_size}
      true -> compute_depth_and_leaf_size(left_size, min_leaf_size, depth + 1)
    end
  end

  defn left_child(node) do
    2 * node + 1
  end

  defn right_child(node) do
    2 * node + 2
  end

  @doc """
  """
  deftransform fit_bounded(tensor, opts) do
    num_trees = opts[:num_trees]
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    size = Nx.axis_size(tensor, 0)
    # TODO: Calculate depth from tensor
    # floor(log2(size / min_leaf_size)) might do the job!
    {depth, leaf_size} = compute_depth_and_leaf_size(size, opts[:min_leaf_size], 0)

    {_leaf_size, indices, hyperplanes, medians} =
      fit_bounded_n(tensor, key, depth: depth, num_trees: num_trees)

    %__MODULE__{
      depth: depth,
      leaf_size: leaf_size,
      num_trees: num_trees,
      indices: indices,
      data: tensor,
      hyperplanes: hyperplanes,
      medians: medians
    }
  end

  defn fit_bounded_n(tensor, key, opts) do
    depth = opts[:depth]
    num_trees = opts[:num_trees]
    {size, dim} = Nx.shape(tensor)

    {hyperplanes, _key} =
      Nx.Random.normal(key, type: to_float_type(tensor), shape: {num_trees, depth, dim})

    num_nodes = 2 ** depth - 1

    {indices, medians, leaf_size, _} =
      while {
              indices = Nx.iota({1, size}, type: :u32) |> Nx.broadcast({num_trees, size}),
              # TODO: Add type for medians!
              medians = Nx.broadcast(:nan, {num_trees, num_nodes}),
              leaf_size = Nx.u32(size),
              {
                tensor,
                hyperplanes,
                level = 0,
                pos = Nx.iota({size}, type: :u32),
                tree_sizes = Nx.broadcast(Nx.u32(size), {size}),
                tags = Nx.broadcast(Nx.u32(0), {size}),
                nodes = Nx.iota({num_nodes}),
                width = Nx.u32(1),
                median_offset = Nx.u32(0)
              }
            },
            level < depth do
        h = hyperplanes[[.., level]]
        proj = Nx.dot(h, [1], tensor, [1]) |> Nx.take_along_axis(indices, axis: 1)

        min = Nx.reduce_min(proj, axes: [1], keep_axes: true)
        max = Nx.reduce_max(proj, axes: [1], keep_axes: true)
        amplitude = Nx.abs(max - min)
        band = amplitude + 1

        sorted_indices = Nx.argsort(band * tags + proj, axis: 1, type: :u32)
        indices = Nx.take_along_axis(indices, sorted_indices, axis: 1)
        proj = Nx.take_along_axis(proj, sorted_indices, axis: 1)

        right_sizes = Nx.quotient(tree_sizes, 2)
        left_sizes = right_sizes + Nx.remainder(tree_sizes, 2)
        tree_sizes = Nx.select(pos < left_sizes, left_sizes, right_sizes)
        tags = 2 * tags + (pos >= tree_sizes)

        leaf_size = Nx.quotient(leaf_size, 2) + Nx.remainder(leaf_size, 2)

        left_mask = (pos == left_sizes - 1) |> Nx.new_axis(0) |> Nx.broadcast({num_trees, size})
        left_or_inf = Nx.select(left_mask, proj, :infinity)

        sorted_indices =
          Nx.argsort(band * tags + left_or_inf, axis: 1, type: :u32)
          |> Nx.slice_along_axis(0, num_nodes, axis: 1)

        left_first = Nx.take_along_axis(left_or_inf, sorted_indices, axis: 1)

        right_mask = (pos == right_sizes) |> Nx.new_axis(0) |> Nx.broadcast({num_trees, size})
        right_or_inf = Nx.select(right_mask, proj, :infinity)

        sorted_indices =
          Nx.argsort(band * tags + right_or_inf, axis: 1, type: :u32)
          |> Nx.slice_along_axis(0, num_nodes, axis: 1)

        right_first = Nx.take_along_axis(right_or_inf, sorted_indices, axis: 1)

        medians_first = (left_first + right_first) / 2

        # TODO: Can this be done in a simpler way?

        # TODO: Write this in a different way so that nodes can be u32
        median_pos =
          Nx.select(width <= nodes and nodes < width + median_offset, -nodes, nodes)
          |> Nx.argsort(type: :u32)

        median_slice = Nx.take(medians_first, median_pos, axis: 1)
        tree_nodes = nodes |> Nx.new_axis(0) |> Nx.broadcast({num_trees, num_nodes})

        medians =
          Nx.select(
            median_offset <= tree_nodes and tree_nodes < median_offset + width,
            median_slice,
            medians
          )

        pos = Nx.remainder(pos, left_sizes)

        {
          indices,
          medians,
          leaf_size,
          {tensor, hyperplanes, level + 1, pos, tree_sizes, tags, nodes, 2 * width,
           2 * median_offset + 1}
        }
      end

    {leaf_size, indices, hyperplanes, medians}
  end
end