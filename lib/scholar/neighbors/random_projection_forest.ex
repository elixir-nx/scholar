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
      iex> Scholar.Neighbors.RandomProjectionForest.fit(tensor, num_trees: 3, min_leaf_size: 2, key: key).indices
      #Nx.Tensor<
        u32[3][5]
        [
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [4, 3, 2, 1, 0]
        ]
      >
  """
  deftransform fit(tensor, opts) do
    if Nx.rank(tensor) != 2 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(tensor))}\
            """
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)
    min_leaf_size = opts[:min_leaf_size]
    num_trees = opts[:num_trees]
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    size = Nx.axis_size(tensor, 0)
    # TODO: Try calculating depth from tensor
    # floor(log2(size / min_leaf_size)) might do the job!
    {depth, leaf_size} = compute_depth_and_leaf_size(size, min_leaf_size, 0)

    if depth == 0 do
      raise ArgumentError,
            """
            expected num_samples to be at least twice \
            min_leaf_size = #{inspect(min_leaf_size)}, got #{inspect(size)}
            """
    end

    {indices, hyperplanes, medians} = fit_n(tensor, key, depth: depth, num_trees: num_trees)

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

  defp compute_depth_and_leaf_size(size, min_leaf_size, depth) do
    right_size = div(size, 2)
    left_size = right_size + rem(size, 2)

    cond do
      right_size < min_leaf_size ->
        {depth, size}

      right_size == min_leaf_size ->
        {depth + 1, left_size}

      true ->
        new_size = if rem(left_size, 2) == 1, do: left_size, else: right_size
        compute_depth_and_leaf_size(new_size, min_leaf_size, depth + 1)
    end
  end

  defn fit_n(tensor, key, opts) do
    depth = opts[:depth]
    num_trees = opts[:num_trees]
    type = to_float_type(tensor)
    {size, dim} = Nx.shape(tensor)
    num_nodes = 2 ** depth - 1

    {hyperplanes, _key} =
      Nx.Random.normal(key, type: type, shape: {num_trees, depth, dim})

    proj = Nx.dot(hyperplanes, [2], tensor, [1])
    sorted_indices = Nx.argsort(proj, axis: 2, stable: true, type: :u32)

    {indices, medians, _} =
      while {
              indices = Nx.iota({num_trees, size}, axis: 1, type: :u32),
              medians = Nx.broadcast(Nx.tensor(:nan, type: type), {num_trees, num_nodes}),
              {
                proj,
                sorted_indices,
                level = Nx.u32(0),
                pos = Nx.iota({size}, type: :u32),
                cell_sizes = Nx.broadcast(Nx.u32(size), {size}),
                tags = Nx.broadcast(Nx.u32(0), {size}),
                nodes = Nx.iota({num_nodes}, type: :u32),
                width = Nx.u32(1),
                median_offset = Nx.u32(0)
              }
            },
            level < depth do
        level_proj = proj[[.., level]] |> Nx.take_along_axis(indices, axis: 1)

        level_indices =
          indices
          |> inverse_permutation()
          |> Nx.take_along_axis(sorted_indices[[.., level]], axis: 1)

        orders = Nx.argsort(tags[level_indices], axis: 1, stable: true, type: :u32)
        level_indices = Nx.take_along_axis(level_indices, orders, axis: 1)
        indices = Nx.take_along_axis(indices, level_indices, axis: 1)
        level_proj = Nx.take_along_axis(level_proj, level_indices, axis: 1)

        right_sizes = Nx.quotient(cell_sizes, 2)
        left_sizes = right_sizes + Nx.remainder(cell_sizes, 2)
        cell_sizes = Nx.select(pos < left_sizes, left_sizes, right_sizes)
        tags = 2 * tags + (pos >= cell_sizes)

        medians =
          update_medians(
            pos,
            left_sizes,
            right_sizes,
            level_proj,
            nodes,
            width,
            median_offset,
            medians
          )

        pos = Nx.remainder(pos, left_sizes)

        {
          indices,
          medians,
          {proj, sorted_indices, level + 1, pos, cell_sizes, tags, nodes, 2 * width,
           2 * median_offset + 1}
        }
      end

    {indices, hyperplanes, medians}
  end

  defnp inverse_permutation(indices) do
    {num_trees, size} = Nx.shape(indices)
    target = Nx.broadcast(Nx.u32(0), {num_trees, size})
    trees = Nx.iota({num_trees, size, 1}, axis: 0)

    indices =
      Nx.concatenate([trees, Nx.new_axis(indices, 2)], axis: 2)
      |> Nx.reshape({num_trees * size, 2})

    updates = Nx.iota({num_trees, size}, axis: 1) |> Nx.reshape({num_trees * size})
    Nx.indexed_add(target, indices, updates)
  end

  defnp update_medians(
          pos,
          left_sizes,
          right_sizes,
          level_proj,
          nodes,
          width,
          median_offset,
          medians
        ) do
    size = Nx.size(pos)
    {num_trees, num_nodes} = Nx.shape(medians)

    left_mask = pos == left_sizes - 1

    left_indices =
      Nx.argsort(left_mask, direction: :desc, stable: true, type: :u32)
      |> Nx.new_axis(0)
      |> Nx.broadcast({num_trees, size})

    left_first = Nx.take_along_axis(level_proj, left_indices, axis: 1)

    right_mask = pos == right_sizes

    right_indices =
      Nx.argsort(right_mask, direction: :desc, stable: true, type: :u32)
      |> Nx.new_axis(0)
      |> Nx.broadcast({num_trees, size})

    right_first = Nx.take_along_axis(level_proj, right_indices, axis: 1)

    medians_first = (left_first + right_first) / 2

    median_mask = width <= nodes and nodes < width + median_offset
    median_pos = Nx.argsort(median_mask, direction: :desc, stable: true, type: :u32)
    level_medians = Nx.take(medians_first, median_pos, axis: 1)

    level_mask =
      (median_offset <= nodes and nodes < median_offset + width)
      |> Nx.new_axis(0)
      |> Nx.broadcast({num_trees, num_nodes})

    Nx.select(
      level_mask,
      level_medians,
      medians
    )
  end

  @doc """
  Computes the leaf indices for every point in the input tensor.
  If the input tensor contains n points, then the result has shape {n, num_trees, leaf_size}.

  ## Examples

      iex> key = Nx.Random.key(12)
      iex> tensor = Nx.iota({5, 2})
      iex> forest = Scholar.Neighbors.RandomProjectionForest.fit(tensor, num_trees: 3, min_leaf_size: 2, key: key)
      iex> x = Nx.tensor([[3, 4]])
      iex> Scholar.Neighbors.RandomProjectionForest.predict(forest, x)
      #Nx.Tensor<
        u32[1][3][3]
        [
          [
            [0, 1, 2],
            [0, 1, 2],
            [4, 3, 2]
          ]
        ]
      >
  """
  deftransform predict(forest, x) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}\
            """
    end

    if Nx.axis_size(forest.hyperplanes, 2) != Nx.axis_size(x, 1) do
      raise ArgumentError,
            """
            expected hyperplanes and input tensor to have the same dimension, \
            got #{inspect(Nx.axis_size(forest.hyperplanes, 2))} \
            and #{inspect(Nx.axis_size(x, 1))}
            """
    end

    predict_n(forest, x)
  end

  defn predict_n(forest, x) do
    num_trees = forest.num_trees
    leaf_size = forest.leaf_size
    indices = forest.indices |> Nx.vectorize(:trees)
    start_indices = compute_start_indices(forest, x, leaf_size: leaf_size) |> Nx.new_axis(1)
    size = Nx.axis_size(x, 0)

    pos =
      Nx.iota({1, 1, leaf_size})
      |> Nx.broadcast({num_trees, size, leaf_size})
      |> Nx.vectorize(:trees)
      |> Nx.add(start_indices)

    Nx.take(indices, pos)
    |> Nx.devectorize()
    |> Nx.rename(nil)
    |> Nx.transpose(axes: [1, 0, 2])
  end

  defn compute_start_indices(forest, x, opts) do
    leaf_size = opts[:leaf_size]
    size = Nx.axis_size(x, 0)
    depth = forest.depth
    num_trees = forest.num_trees
    hyperplanes = forest.hyperplanes |> Nx.vectorize(:trees)
    medians = forest.medians |> Nx.vectorize(:trees)

    {start_indices, left?, cell_sizes, _} =
      while {
              start_indices = Nx.broadcast(Nx.u32(0), {num_trees, size}) |> Nx.vectorize(:trees),
              left? = Nx.broadcast(Nx.u8(0), {num_trees, size}) |> Nx.vectorize(:trees),
              cell_sizes = Nx.broadcast(Nx.u32(size), {num_trees, size}) |> Nx.vectorize(:trees),
              {
                x,
                hyperplanes,
                medians,
                level = 0,
                nodes = Nx.broadcast(Nx.u32(0), {num_trees, size}) |> Nx.vectorize(:trees)
              }
            },
            level < depth do
        h = hyperplanes[level]
        median = Nx.take(medians, nodes)
        proj = Nx.dot(x, h)
        left? = proj <= median

        nodes =
          Nx.select(
            left?,
            left_child(nodes),
            right_child(nodes)
          )

        right_sizes = Nx.quotient(cell_sizes, 2)
        left_sizes = right_sizes + Nx.remainder(cell_sizes, 2)
        start_indices = Nx.select(left?, start_indices, start_indices + left_sizes)
        cell_sizes = Nx.select(left?, left_sizes, right_sizes)

        {
          start_indices,
          left?,
          cell_sizes,
          {x, hyperplanes, medians, level + 1, nodes}
        }
      end

    Nx.select(not left? and cell_sizes < leaf_size, start_indices - 1, start_indices)
  end

  defn left_child(nodes) do
    2 * nodes + 1
  end

  defn right_child(nodes) do
    2 * nodes + 2
  end
end
