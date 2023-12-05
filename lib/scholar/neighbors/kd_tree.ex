defmodule Scholar.Neighbors.KDTree do
  @moduledoc """
  Implements a kd-tree, a space-partitioning data structure for organizing points
  in a k-dimensional space.

  This is implemented as one-dimensional tensor with indices pointed to highest
  dimension of the given tensor. Traversal starts by calling `root/0` and then
  accessing the `left_child/1` and `right_child/1`. The tree is left-balanced.

  Two construction modes are available:

    * `fit_bounded/2` - the tensor has min and max values with an amplitude given by `max - min`.
      It is also guaranteed that the `amplitude * levels(tensor) + 1` does not overflow
      the tensor. See `amplitude/1` to verify if this holds. This implementation happens
      fully within `defn`. This version is orders of magnitude faster than the `unbounded/2`
      one.

    * `fit_unbounded/2` - there are no known bounds (min and max values) to the tensor.
      This implementation is recursive and goes in and out of the `defn`, therefore
      it cannot be called inside `defn`.

  Each level traverses over the last axis of tensor, the index for a level can be
  computed as: `rem(level, Nx.axis_size(tensor, -1))`.

  ## References

    * [GPU-friendly, Parallel, and (Almost-)In-Place Construction of Left-Balanced k-d Trees](https://arxiv.org/pdf/2211.00120.pdf).
  """

  import Nx.Defn
  alias Scholar.Metrics.Distance

  @derive {Nx.Container, keep: [:levels], containers: [:indices, :data]}
  @enforce_keys [:levels, :indices, :data]
  defstruct [:levels, :indices, :data]

  opts = [
    k: [
      type: :pos_integer,
      default: 3,
      doc: "The number of neighbors to use by default for `k_neighbors` queries"
    ],
    metric: [
      type: {:custom, Scholar.Options, :metric, []},
      default: {:minkowski, 2},
      doc: ~S"""
      Name of the metric. Possible values:

      * `{:minkowski, p}` - Minkowski metric. By changing value of `p` parameter (a positive number or `:infinity`)
        we can set Manhattan (`1`), Euclidean (`2`), Chebyshev (`:infinity`), or any arbitrary $L_p$ metric.

      * `:cosine` - Cosine metric.
      """
    ]
  ]

  @predict_schema NimbleOptions.new!(opts)

  @doc """
  Builds a KDTree without known min-max bounds.

  If your tensor has known bounds (for example, -1 and 1),
  consider using the `bounded/2` version which is often orders of
  magnitude more efficient.

  ## Options

    * `:compiler` - the default compiler to use for internal defn operations

  ## Examples

      iex> Scholar.Neighbors.KDTree.fit_unbounded(Nx.iota({5, 2}), compiler: EXLA)
      %Scholar.Neighbors.KDTree{
        data: Nx.iota({5, 2}),
        levels: 3,
        indices: Nx.u32([3, 1, 4, 0, 2])
      }

  """
  def fit_unbounded(tensor, opts \\ []) do
    levels = levels(tensor)
    {size, _dims} = Nx.shape(tensor)

    indices =
      if size > 2 do
        subtree_size = unbounded_subtree_size(1, levels, size)
        {left, mid, right} = Nx.Defn.jit_apply(&root_slice(&1, subtree_size), [tensor], opts)

        acc = <<Nx.to_number(mid)::32-unsigned-native-integer>>
        acc = recur([{1, left}, {2, right}], [], acc, tensor, 1, levels, opts)
        Nx.from_binary(acc, :u32)
      else
        Nx.argsort(tensor[[.., 0]], direction: :desc, type: :u32)
      end

    %__MODULE__{levels: levels, indices: indices, data: tensor}
  end

  defp recur([{_i, %Nx.Tensor{shape: {1}} = leaf} | rest], next, acc, tensor, level, levels, opts) do
    [leaf] = Nx.to_flat_list(leaf)
    acc = <<acc::binary, leaf::32-unsigned-native-integer>>
    recur(rest, next, acc, tensor, level, levels, opts)
  end

  defp recur([{i, %Nx.Tensor{shape: {2}} = node} | rest], next, acc, tensor, level, levels, opts) do
    acc = <<acc::binary, Nx.to_number(node[1])::32-unsigned-native-integer>>
    next = [{left_child(i), Nx.slice(node, [0], [1])} | next]
    recur(rest, next, acc, tensor, level, levels, opts)
  end

  defp recur([{i, indices} | rest], next, acc, tensor, level, levels, opts) do
    %Nx.Tensor{shape: {size, dims}} = tensor
    k = rem(level, dims)
    subtree_size = unbounded_subtree_size(left_child(i), levels, size)

    {left, mid, right} =
      Nx.Defn.jit_apply(&recur_slice(&1, &2, &3, subtree_size), [tensor, indices, k], opts)

    next = [{right_child(i), right}, {left_child(i), left} | next]
    acc = <<acc::binary, Nx.to_number(mid)::32-unsigned-native-integer>>
    recur(rest, next, acc, tensor, level, levels, opts)
  end

  defp recur([], [], acc, _tensor, _level, _levels, _opts) do
    acc
  end

  defp recur([], next, acc, tensor, level, levels, opts) do
    recur(Enum.reverse(next), [], acc, tensor, level + 1, levels, opts)
  end

  defp root_slice(tensor, subtree_size) do
    indices = Nx.argsort(tensor[[.., 0]], type: :u32)

    {Nx.slice(indices, [0], [subtree_size]), indices[subtree_size],
     Nx.slice(indices, [subtree_size + 1], [Nx.size(indices) - subtree_size - 1])}
  end

  defp recur_slice(tensor, indices, k, subtree_size) do
    sorted = Nx.argsort(Nx.take(tensor, indices)[[.., k]], type: :u32)
    indices = Nx.take(indices, sorted)

    {Nx.slice(indices, [0], [subtree_size]), indices[subtree_size],
     Nx.slice(indices, [subtree_size + 1], [Nx.size(indices) - subtree_size - 1])}
  end

  defp unbounded_subtree_size(i, levels, size) do
    import Bitwise
    diff = levels - unbounded_level(i) - 1
    shifted = 1 <<< diff
    fllc_s = (i <<< diff) + shifted - 1
    shifted - 1 + min(max(0, size - fllc_s), shifted)
  end

  defp unbounded_level(i) when is_integer(i), do: floor(:math.log2(i + 1))

  @doc """
  Builds a KDTree with known min-max bounds entirely within `defn`.

  This requires the amplitude `|max - min|` of the tensor to be given
  such that `max + (amplitude + 1) * (size - 1)` does not overflow the
  maximum tensor type.

  For example, a tensor where all values are between 0 and 1 has amplitude
  1. Values between -1 and 1 has amplitude 2. If your tensor is normalized
  to floating points, then it is most likely bounded (given their high
  precision). You can use `amplitude/1` to check your assumptions.

  ## Examples

      iex> Scholar.Neighbors.KDTree.fit_bounded(Nx.iota({5, 2}), 10)
      %Scholar.Neighbors.KDTree{
        data: Nx.iota({5, 2}),
        levels: 3,
        indices: Nx.u32([3, 1, 4, 0, 2])
      }
  """
  deftransform fit_bounded(tensor, amplitude) do
    %__MODULE__{levels: levels(tensor), indices: fit_bounded_n(tensor, amplitude), data: tensor}
  end

  defnp fit_bounded_n(tensor, amplitude) do
    levels = levels(tensor)
    {size, dims} = Nx.shape(tensor)
    band = amplitude + 1
    tags = Nx.broadcast(Nx.u32(0), {size})

    {level, tags, _tensor, _band} =
      while {level = Nx.u32(0), tags, tensor, band}, level < levels - 1 do
        k = rem(level, dims)
        indices = Nx.argsort(tensor[[.., k]] + band * tags, type: :u32)
        tags = update_tags(tags, indices, level, levels, size)
        {level + 1, tags, tensor, band}
      end

    k = rem(level, dims)
    Nx.argsort(tensor[[.., k]] + band * tags, type: :u32)
  end

  defnp update_tags(tags, indices, level, levels, size) do
    pos = Nx.argsort(indices, type: :u32)

    pivot =
      bounded_segment_begin(tags, levels, size) +
        bounded_subtree_size(left_child(tags), levels, size)

    Nx.select(
      pos < (1 <<< level) - 1,
      tags,
      Nx.select(
        pos < pivot,
        left_child(tags),
        Nx.select(
          pos > pivot,
          right_child(tags),
          tags
        )
      )
    )
  end

  defnp bounded_subtree_size(i, levels, size) do
    diff = levels - bounded_level(i) - 1
    shifted = 1 <<< diff
    first_lowest_level = (i <<< diff) + shifted - 1
    # Use select instead of max to deal with overflows
    lowest_level = Nx.select(first_lowest_level > size, Nx.u32(0), size - first_lowest_level)
    shifted - 1 + min(lowest_level, shifted)
  end

  defnp bounded_segment_begin(i, levels, size) do
    level = bounded_level(i)
    top = (1 <<< level) - 1
    diff = levels - level - 1
    shifted = 1 <<< diff
    left_siblings = i - top

    top + left_siblings * (shifted - 1) +
      min(left_siblings * shifted, size - (1 <<< (levels - 1)) + 1)
  end

  # Since this property relies on u32, let's check the tensor type.
  deftransformp bounded_level(%Nx.Tensor{type: {:u, 32}} = i) do
    Nx.subtract(31, Nx.count_leading_zeros(Nx.add(i, 1)))
  end

  @doc """
  Returns the amplitude of a bounded tensor.

  If -1 is returned, it means the tensor cannot use the `fit_bounded` algorithm
  to generate a KDTree and `fit_unbounded/2` must be used instead.

  This cannot be invoked inside a `defn`.

  ## Examples

      iex> Scholar.Neighbors.KDTree.amplitude(Nx.iota({10, 2}))
      19
      iex> Scholar.Neighbors.KDTree.amplitude(Nx.iota({20, 2}, type: :f32))
      39.0
      iex> Scholar.Neighbors.KDTree.amplitude(Nx.iota({20, 2}, type: :u8))
      -1
      iex> Scholar.Neighbors.KDTree.amplitude(Nx.negate(Nx.iota({10, 2})))
      19

  """
  def amplitude(tensor) do
    max = tensor |> Nx.reduce_max() |> Nx.to_number()
    min = tensor |> Nx.reduce_min() |> Nx.to_number()
    amplitude = abs(max - min)
    limit = tensor.type |> Nx.Constants.max_finite() |> Nx.to_number()

    if max + (amplitude + 1) * (Nx.axis_size(tensor, 0) - 1) > limit do
      -1
    else
      amplitude
    end
  end

  @doc """
  Returns the number of resulting levels in a KDTree for `tensor`.

  ## Examples

      iex> Scholar.Neighbors.KDTree.levels(Nx.iota({10, 3}))
      4
  """
  deftransform levels(%Nx.Tensor{} = tensor) do
    case Nx.shape(tensor) do
      {size, _dims} -> ceil(:math.log2(size + 1))
      _ -> raise ArgumentError, "KDTrees requires a tensor of rank 2"
    end
  end

  @doc """
  Returns the root index.

  ## Examples

      iex> Scholar.Neighbors.KDTree.root()
      0

  """
  deftransform root, do: 0

  @doc """
  Returns the parent of child `i`.

  It is your responsibility to guarantee the result is positive.

  ## Examples

      iex> Scholar.Neighbors.KDTree.parent(1)
      0
      iex> Scholar.Neighbors.KDTree.parent(2)
      0

      iex> Scholar.Neighbors.KDTree.parent(Nx.u32(3))
      #Nx.Tensor<
        u32
        1
      >

  """
  deftransform parent(0) do
    -1
  end

  deftransform parent(i) when is_integer(i), do: div(i - 1, 2)
  deftransform parent(%Nx.Tensor{} = t), do: Nx.quotient(Nx.subtract(t, 1), 2)

  @doc """
  Returns the index of the left child of i.

  It is your responsibility to guarantee the result
  is not greater than the leading axis of the tensor.

  ## Examples

      iex> Scholar.Neighbors.KDTree.left_child(0)
      1
      iex> Scholar.Neighbors.KDTree.left_child(1)
      3

      iex> Scholar.Neighbors.KDTree.left_child(Nx.u32(3))
      #Nx.Tensor<
        u32
        7
      >

  """
  deftransform left_child(i) when is_integer(i), do: 2 * i + 1
  deftransform left_child(%Nx.Tensor{} = t), do: Nx.add(Nx.multiply(2, t), 1)

  @doc """
  Returns the index of the right child of i.

  It is your responsibility to guarantee the result
  is not greater than the leading axis of the tensor.

  ## Examples

      iex> Scholar.Neighbors.KDTree.right_child(0)
      2
      iex> Scholar.Neighbors.KDTree.right_child(1)
      4

      iex> Scholar.Neighbors.KDTree.right_child(Nx.u32(3))
      #Nx.Tensor<
        u32
        8
      >

  """
  deftransform right_child(i) when is_integer(i), do: 2 * i + 2
  deftransform right_child(%Nx.Tensor{} = t), do: Nx.add(Nx.multiply(2, t), 2)

  @doc """
  Predict the K nearest neighbors of `x_predict` in KDTree.

  ## Examples

      iex> x = Nx.iota({10, 2})
      iex> x_predict = Nx.tensor([[2, 5], [1, 9], [6, 4]])
      iex> kdtree = Scholar.Neighbors.KDTree.fit_bounded(x, 20)
      iex> Scholar.Neighbors.KDTree.predict(kdtree, x_predict, k: 3)
      #Nx.Tensor<
        s64[3][3]
        [
          [2, 1, 0],
          [2, 3, 1],
          [2, 3, 1]
        ]
      >
      iex> Scholar.Neighbors.KDTree.predict(kdtree, x_predict, k: 3, metric: {:minkowski, 1})
      #Nx.Tensor<
        s64[3][3]
        [
          [2, 1, 0],
          [4, 3, 1],
          [2, 3, 1]
        ]
      >
  """
  deftransform predict(tree, data, opts \\ []) do
    predict_n(tree, data, NimbleOptions.validate!(opts, @predict_schema))
  end

  defnp predict_n(tree, data, opts) do
    query_points(data, tree, opts) |> Nx.devectorize(keep_names: false)
  end

  defnp sort_by_distances(distances, point_indices) do
    indices = Nx.argsort(distances)
    {Nx.take(distances, indices), Nx.take(point_indices, indices)}
  end

  defnp compute_distance(x1, x2, opts) do
    case opts[:metric] do
      {:minkowski, 2} -> Distance.squared_euclidean(x1, x2)
      {:minkowski, p} -> Distance.minkowski(x1, x2, p: p)
      :cosine -> Distance.cosine(x1, x2)
    end
  end

  defnp update_knn(nearest_neighbors, distances, data, indices, curr_node, point, k, opts) do
    curr_dist = compute_distance(data[[indices[curr_node]]], point, opts)

    if curr_dist < distances[[-1]] do
      nearest_neighbors =
        Nx.indexed_put(nearest_neighbors, Nx.new_axis(k - 1, 0), indices[curr_node])

      distances = Nx.indexed_put(distances, Nx.new_axis(k - 1, 0), curr_dist)
      sort_by_distances(distances, nearest_neighbors)
    else
      {distances, nearest_neighbors}
    end
  end

  defnp update_visited(node, visited, distances, nearest_neighbors, data, indices, point, k, opts) do
    if visited[indices[node]] do
      {visited, {distances, nearest_neighbors}}
    else
      visited = Nx.indexed_put(visited, Nx.new_axis(indices[node], 0), Nx.u8(1))

      {distances, nearest_neighbors} =
        update_knn(nearest_neighbors, distances, data, indices, node, point, k, opts)

      {visited, {distances, nearest_neighbors}}
    end
  end

  defnp query_points(point, tree, opts) do
    k = opts[:k]
    node = Nx.as_type(root(), :s64)
    point = point |> Nx.vectorize(:x)
    {size, dims} = Nx.shape(tree.data)
    nearest_neighbors = Nx.broadcast(Nx.s64(0), {k})
    distances = Nx.broadcast(Nx.Constants.infinity(), {k})
    visited = Nx.broadcast(Nx.s64(0), {size})

    indices = tree.indices |> Nx.as_type(:s64)
    data = tree.data

    down = 0
    up = 1
    mode = down
    i = Nx.s64(0)

    [nearest_neighbors, node, distances, visited, i, mode, point] =
      Nx.broadcast_vectors([
        nearest_neighbors,
        node,
        distances,
        visited,
        i,
        mode,
        point
      ])

    {nearest_neighbors, _} =
      while {nearest_neighbors, {node, data, indices, point, distances, visited, i, mode}},
            node != -1 and i >= 0 do
        coord_indicator = rem(i, dims)

        {node, i, visited, nearest_neighbors, distances, mode} =
          cond do
            node >= size ->
              {parent(node), i - 1, visited, nearest_neighbors, distances, up}

            mode == down and
                point[[coord_indicator]] < data[[indices[node], coord_indicator]] ->
              {left_child(node), i + 1, visited, nearest_neighbors, distances, down}

            mode == down and
                point[[coord_indicator]] >= data[[indices[node], coord_indicator]] ->
              {right_child(node), i + 1, visited, nearest_neighbors, distances, down}

            mode == up ->
              cond do
                visited[indices[node]] ->
                  {parent(node), i - 1, visited, nearest_neighbors, distances, up}

                (left_child(node) >= size and right_child(node) >= size) or
                  (left_child(node) < size and visited[indices[left_child(node)]] and
                     right_child(node) < size and
                     visited[indices[right_child(node)]]) or
                    (left_child(node) < size and visited[indices[left_child(node)]] and
                       right_child(node) >= size) ->
                  {visited, {distances, nearest_neighbors}} =
                    update_visited(
                      node,
                      visited,
                      distances,
                      nearest_neighbors,
                      data,
                      indices,
                      point,
                      k,
                      opts
                    )

                  {parent(node), i - 1, visited, nearest_neighbors, distances, up}

                left_child(node) < size and visited[indices[left_child(node)]] and
                  right_child(node) < size and
                    not visited[indices[right_child(node)]] ->
                  {visited, {distances, nearest_neighbors}} =
                    update_visited(
                      node,
                      visited,
                      distances,
                      nearest_neighbors,
                      data,
                      indices,
                      point,
                      k,
                      opts
                    )

                  if Nx.any(
                       compute_distance(
                         point[[coord_indicator]],
                         data[[indices[right_child(node)], coord_indicator]],
                         opts
                       ) <
                         distances
                     ) do
                    {right_child(node), i + 1, visited, nearest_neighbors, distances, down}
                  else
                    {parent(node), i - 1, visited, nearest_neighbors, distances, up}
                  end

                ((right_child(node) < size and visited[indices[right_child(node)]]) or
                   right_child(node) == size) and
                    not visited[indices[left_child(node)]] ->
                  {visited, {distances, nearest_neighbors}} =
                    update_visited(
                      node,
                      visited,
                      distances,
                      nearest_neighbors,
                      data,
                      indices,
                      point,
                      k,
                      opts
                    )

                  if Nx.any(
                       compute_distance(
                         point[[coord_indicator]],
                         data[[indices[left_child(node)], coord_indicator]],
                         opts
                       ) <
                         distances
                     ) do
                    {left_child(node), i + 1, visited, nearest_neighbors, distances, down}
                  else
                    {parent(node), i - 1, visited, nearest_neighbors, distances, up}
                  end

                # Should be not reachable
                true ->
                  {node, i, visited, nearest_neighbors, distances, mode}
              end

            # Should be not reachable
            true ->
              {node, i, visited, nearest_neighbors, distances, mode}
          end

        {nearest_neighbors, {node, data, indices, point, distances, visited, i, mode}}
      end

    nearest_neighbors
  end

  defn factorial(n) do
    {_, res, _} =
      while {i = 1, res = 1, n}, i < n do
        res = res * i
        {i + 1, res, n}
      end

    res
  end

  defn general_factorial(x) do
    x = Nx.vectorize(x, :x)
    factorial(x)
  end
end
