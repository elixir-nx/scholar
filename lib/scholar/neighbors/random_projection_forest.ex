defmodule Scholar.Neighbors.RandomProjectionForest do
  @moduledoc """
  Random Projection Forest for k-Nearest Neighbor Search.

  Each tree in a forest is constructed using a divide and conquer approach.
  We start with the entire dataset and at every node we project the data onto a random
  hyperplane and split it in the following way: the points with the projection smaller
  than or equal to the median are put into the left subtree and the points with projection
  greater than the median are put into the right subtree. We then proceed
  recursively with the left and right subtree.

  In this implementation the trees are complete, i.e. there are 2^l nodes at level l.
  The leaves of the trees are arranged as blocks in the field `indices`. We use the same
  hyperplane for all nodes on the same level as in [2].

  * [1] - Randomized partition trees for nearest neighbor search
  * [2] - Fast Nearest Neighbor Search through Sparse Random Projections and Voting
  """

  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Neighbors.Utils

  @derive {Nx.Container,
           keep: [:num_neighbors, :metric, :depth, :leaf_size, :num_trees],
           containers: [:indices, :data, :hyperplanes, :medians]}
  @enforce_keys [
    :num_neighbors,
    :depth,
    :leaf_size,
    :num_trees,
    :indices,
    :data,
    :hyperplanes,
    :medians
  ]
  defstruct [
    :num_neighbors,
    :metric,
    :depth,
    :leaf_size,
    :num_trees,
    :indices,
    :data,
    :hyperplanes,
    :medians
  ]

  opts = [
    num_neighbors: [
      required: true,
      type: :pos_integer,
      doc: "The number of nearest neighbors."
    ],
    metric: [
      type: {:in, [:squared_euclidean, :euclidean]},
      default: :euclidean,
      doc: "The function that measures the distance between two points."
    ],
    min_leaf_size: [
      type: :pos_integer,
      doc: "The minumum number of points in the leaf."
    ],
    num_trees: [
      required: true,
      type: :pos_integer,
      doc: "The number of trees in the forest."
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
      iex> forest = Scholar.Neighbors.RandomProjectionForest.fit(tensor, num_neighbors: 2, num_trees: 3, key: key)
      iex> forest.indices
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
    num_neighbors = opts[:num_neighbors]
    min_leaf_size = opts[:min_leaf_size]

    metric =
      case opts[:metric] do
        :euclidean -> &Scholar.Metrics.Distance.euclidean/2
        :squared_euclidean -> &Scholar.Metrics.Distance.squared_euclidean/2
      end

    min_leaf_size =
      cond do
        is_nil(min_leaf_size) ->
          num_neighbors

        min_leaf_size >= num_neighbors ->
          min_leaf_size

        true ->
          raise ArgumentError,
                """
                expected min_leaf_size to be at least num_neighbors = #{inspect(num_neighbors)}, \
                got #{inspect(min_leaf_size)}
                """
      end

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
      num_neighbors: num_neighbors,
      metric: metric,
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

    {indices, medians, _} =
      while {
              indices = Nx.iota({num_trees, size}, axis: 1, type: :u32),
              medians = Nx.broadcast(Nx.tensor(:nan, type: type), {num_trees, num_nodes}),
              {
                tensor,
                hyperplanes,
                level = Nx.u32(0),
                pos = Nx.iota({size}, type: :u32),
                cell_sizes = Nx.broadcast(Nx.u32(size), {size}),
                tags = Nx.broadcast(Nx.u32(0), {size}),
                width = Nx.u32(1),
                median_offset = Nx.u32(0)
              }
            },
            level < depth do
        level_proj =
          Nx.dot(hyperplanes[[.., level]], [1], tensor, [1])
          |> Nx.take_along_axis(indices, axis: 1)

        level_indices = Nx.argsort(level_proj, axis: 1, type: :u32, stable: true)
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
            width,
            median_offset,
            medians
          )

        pos = Nx.remainder(pos, left_sizes)

        {
          indices,
          medians,
          {tensor, hyperplanes, level + 1, pos, cell_sizes, tags, 2 * width,
           2 * median_offset + 1}
        }
      end

    {indices, hyperplanes, medians}
  end

  defnp update_medians(pos, left_sizes, right_sizes, level_proj, width, median_offset, medians) do
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

    nodes = Nx.iota({num_nodes}, type: :u32)
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
  Computes approximate nearest neighbors of query tensor using random projection forest.
  Returns the neighbor indices and distances from query points.

  ## Examples

      iex> key = Nx.Random.key(12)
      iex> tensor = Nx.iota({5, 2})
      iex> forest = Scholar.Neighbors.RandomProjectionForest.fit(tensor, num_neighbors: 2, metric: :squared_euclidean, num_trees: 3, key: key)
      iex> query = Nx.tensor([[3, 4]])
      iex> {neighbors, distances} = Scholar.Neighbors.RandomProjectionForest.predict(forest, query)
      iex> neighbors
      #Nx.Tensor<
        u32[1][2]
        [
          [1, 2]
        ]
      >
      iex> distances
      #Nx.Tensor<
        f32[1][2]
        [
          [2.0, 2.0]
        ]
      >
  """
  deftransform predict(%__MODULE__{} = forest, query) do
    if Nx.rank(query) != 2 do
      raise ArgumentError,
            """
            expected query tensor to have shape {num_queries, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(query))}
            """
    end

    if Nx.axis_size(forest.data, 1) != Nx.axis_size(query, 1) do
      raise ArgumentError,
            """
            expected query tensor to have same number of features as tensor used to grow the forest, \
            got #{inspect(Nx.axis_size(query, 1))} \
            and #{inspect(Nx.axis_size(forest.data, 1))}
            """
    end

    predict_n(forest, query)
  end

  defnp predict_n(forest, query) do
    candidate_indices = get_leaves(forest, query)

    Utils.brute_force_search_with_candidates(forest.data, query, candidate_indices,
      num_neighbors: forest.num_neighbors,
      metric: forest.metric
    )
  end

  @doc false
  defn get_leaves(forest, query) do
    num_trees = forest.num_trees
    leaf_size = forest.leaf_size
    indices = forest.indices |> Nx.vectorize(:trees)
    start_indices = compute_start_indices(forest, query) |> Nx.new_axis(1)
    query_size = Nx.axis_size(query, 0)

    pos =
      Nx.iota({1, 1, leaf_size})
      |> Nx.broadcast({num_trees, query_size, leaf_size})
      |> Nx.vectorize(:trees)
      |> Nx.add(start_indices)

    Nx.take(indices, pos)
    |> Nx.devectorize()
    |> Nx.rename(nil)
    |> Nx.transpose(axes: [1, 0, 2])
    |> Nx.reshape({query_size, num_trees * leaf_size})
  end

  defnp compute_start_indices(forest, query) do
    depth = forest.depth
    leaf_size = forest.leaf_size
    num_trees = forest.num_trees
    hyperplanes = forest.hyperplanes
    medians = forest.medians |> Nx.vectorize(:trees)
    size = Nx.axis_size(forest.data, 0)
    query_size = Nx.axis_size(query, 0)

    {start_indices, left?, cell_sizes, _} =
      while {
              start_indices =
                Nx.broadcast(Nx.u32(0), {num_trees, query_size}) |> Nx.vectorize(:trees),
              _left? = Nx.broadcast(Nx.u8(0), {num_trees, query_size}) |> Nx.vectorize(:trees),
              cell_sizes =
                Nx.broadcast(Nx.u32(size), {num_trees, query_size}) |> Nx.vectorize(:trees),
              {
                query,
                hyperplanes,
                medians,
                level = 0,
                nodes = Nx.broadcast(Nx.u32(0), {num_trees, query_size}) |> Nx.vectorize(:trees)
              }
            },
            level < depth do
        proj = Nx.dot(hyperplanes[[.., level]], [1], query, [1]) |> Nx.vectorize(:trees)
        median = Nx.take(medians, nodes)
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
          {query, hyperplanes, medians, level + 1, nodes}
        }
      end

    Nx.select(not left? and cell_sizes < leaf_size, start_indices - 1, start_indices)
  end

  defnp left_child(nodes), do: 2 * nodes + 1

  defnp right_child(nodes), do: 2 * nodes + 2
end
