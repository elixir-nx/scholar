defmodule Scholar.Neighbors.NNDescent do
  @moduledoc """
  Nearest Neighbors Descent (NND) is an algorithm that calculates Approximated Nearest Neighbors (ANN)
  for a given set of points[1].

  It is implemented using Random Projection Forest[2].

  ## References

    [1] [Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures](https://www.cs.princeton.edu/cass/papers/www11.pdf).
    [2] Random projection trees and low dimensional manifolds
  """

  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Metrics.Distance

  @derive {Nx.Container, containers: [:nearest_neighbors, :distances]}
  defstruct [:nearest_neighbors, :distances]

  opts = [
    num_neighbors: [
      type: :pos_integer,
      required: true,
      doc: "The number of neighbors to use in queries"
    ],
    max_candidates: [
      type: :pos_integer,
      default: 60,
      doc: "The maximum number of candidate neighbors to consider"
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Determines random number generation for centroid initialization.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ],
    max_iterations: [
      type: :pos_integer,
      default: 300,
      doc: "Maximum number of iterations of the k-means algorithm for a single run."
    ],
    tol: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-4,
      doc: """
      Relative tolerance with regards to Frobenius norm of the difference in
      the cluster centers of two consecutive iterations to declare convergence.
      """
    ],
    tree_init?: [
      type: :boolean,
      default: true,
      doc: "Whether to use the tree initialization."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Calculates the approximate nearest neighbors for a given set of points. It
  returns a struct containing two tensors:

  * `nearest_neighbors` - the indices of the nearest neighbors
  * `distances` - the distances to the nearest neighbors

  ## Examples

      iex> data = Nx.iota({10, 5})
      iex> key = Nx.Random.key(12)
      iex> Scholar.Neighbors.NNDescent.fit(data, num_neighbors: 3, key: key)
      %Scholar.Neighbors.NNDescent{
        nearest_neighbors: Nx.tensor(
          [
            [0, 1, 2],
            [1, 2, 0],
            [2, 1, 3],
            [3, 4, 2],
            [4, 5, 3],
            [5, 6, 4],
            [6, 7, 5],
            [7, 6, 8],
            [8, 9, 7],
            [9, 8, 7]
          ], type: :s64
        ),
        distances: Nx.tensor(
          [
            [0.0, 125.0, 500.0],
            [0.0, 125.0, 125.0],
            [0.0, 125.0, 125.0],
            [0.0, 125.0, 125.0],
            [0.0, 125.0, 125.0],
            [0.0, 125.0, 125.0],
            [0.0, 125.0, 125.0],
            [0.0, 125.0, 125.0],
            [0.0, 125.0, 125.0],
            [0.0, 125.0, 500.0]
          ], type: :f32
        )
      }
  """
  deftransform fit(tensor, opts \\ []) do
    opts =
      if Keyword.has_key?(opts, :max_candidates) do
        opts
      else
        Keyword.put(opts, :max_candidates, min(60, opts[:num_neighbors]))
      end

    opts = NimbleOptions.validate!(opts, @opts_schema)
    sum_samples = Nx.axis_size(tensor, 0)

    if opts[:num_neighbors] > sum_samples do
      raise ArgumentError,
            """
            expected num_neighbors to be less than or equal to the number of samples, \
            got num_neighbors: #{opts[:num_neighbors]} and number of samples: \
            #{sum_samples}
            """
    end

    if opts[:max_candidates] > sum_samples do
      raise ArgumentError,
            """
            expected max_candidates to be less than or equal to the number of samples, \
            got max_candidates: #{opts[:max_candidates]} and number of samples: \
            #{sum_samples}
            """
    end

    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    graph = initialize_graph(tensor, opts)

    graph =
      if opts[:tree_init?] do
        leaves =
          get_leaves_from_forest(
            Scholar.Neighbors.RandomProjectionForest.fit(tensor,
              key: key,
              num_trees: min(32, 5 + Kernel.round(:math.pow(Nx.axis_size(tensor, 0), 0.25))),
              min_leaf_size: min(div(sum_samples, 2), max(opts[:num_neighbors], 10)),
              num_neighbors: opts[:num_neighbors]
            )
          )

        update_by_leaves(tensor, graph, leaves)
      else
        graph
      end

    opts = Keyword.put(opts, :max_candidates, min(opts[:max_candidates], opts[:num_neighbors]))
    {graph, key} = init_random(tensor, graph, key, opts)
    nn_descent(tensor, graph, key, opts)
  end

  # Initializes the graph. It returns a tuple of three tensors:

  # * `indices` - the indices of the neighbors
  # * `keys` - the distances to the neighbors
  # * `flags` - flags indicating whether the neighbor is a new candidate or an old candidate
  defnp initialize_graph(tensor, opts) do
    num_neighbors = opts[:num_neighbors]
    num_samples = Nx.axis_size(tensor, 0)

    {Nx.broadcast(Nx.s64(-1), {num_samples, num_neighbors}),
     Nx.broadcast(Nx.Constants.max_finite(to_float_type(tensor)), {num_samples, num_neighbors}),
     Nx.broadcast(Nx.s8(1), {num_samples, num_neighbors})}
  end

  # Returns the size of the heap for each row in the indices tensor.
  # By size of the heap we mean the number of indices that are not -1.
  defnp get_size_vectorized(indices, num_neighbors) do
    indices =
      Nx.concatenate(
        [indices == Nx.s64(-1), Nx.broadcast(Nx.u8(1), {Nx.axis_size(indices, 0), 1})],
        axis: 1
      )

    num_neighbors - Nx.argmax(indices, axis: 1)
  end

  # Initializes the graph with random neighbors.
  defnp init_random(data, curr_graph, key, opts) do
    num_neighbors = opts[:num_neighbors]
    {indices, keys, flags} = curr_graph
    num_heaps = Nx.axis_size(indices, 0)

    missing = get_size_vectorized(indices, num_neighbors)

    {random_indices, new_key} =
      Nx.Random.randint(key, 0, num_heaps, type: :s64, shape: {num_heaps * num_neighbors})

    d =
      Distance.squared_euclidean(
        Nx.reshape(
          Nx.broadcast(Nx.new_axis(data, 1), {num_heaps, num_neighbors, Nx.axis_size(data, -1)}),
          {num_heaps * num_neighbors, Nx.axis_size(data, -1)}
        ),
        Nx.take(data, random_indices),
        axes: [1]
      )

    {{indices, keys, flags}, _} =
      while {{indices, keys, flags}, {index0 = Nx.s64(0), data, missing, random_indices, d}},
            index0 < num_heaps do
        {{indices, keys, flags}, _} =
          while {{indices, keys, flags},
                 {index0, j = Nx.s64(0), data, missing, random_indices, d}},
                j < missing[index0] do
            {add_neighbor(
               {indices, keys, flags},
               index0,
               random_indices[index0 * num_neighbors + j],
               d[index0 * num_neighbors + j],
               Nx.s8(1)
             ), {index0, j + 1, data, missing, random_indices, d}}
          end

        {{indices, keys, flags}, {index0 + 1, data, missing, random_indices, d}}
      end

    {{indices, keys, flags}, new_key}
  end

  # Adds node as its own neighbor with distance 0.
  defnp add_zero_nodes(curr_graph) do
    {indices, keys, flags} = curr_graph
    {num_heaps, num_nodes} = Nx.shape(indices)

    {indices_vectorized_axes, keys_vectorized_axes, flags_vectorized_axes} =
      {indices.vectorized_axes, keys.vectorized_axes, flags.vectorized_axes}

    iota = Nx.revectorize(Nx.iota({num_heaps}), [x: :auto], target_shape: {})

    indices = Nx.revectorize(indices, [x: num_heaps], target_shape: {Nx.axis_size(indices, -1)})
    keys = Nx.revectorize(keys, [x: num_heaps], target_shape: {Nx.axis_size(keys, -1)})
    flags = Nx.revectorize(flags, [x: num_heaps], target_shape: {Nx.axis_size(flags, -1)})

    [iota, zero_node, new_flag, indices, keys, flags] =
      Nx.broadcast_vectors([
        iota,
        Nx.tensor(0.0, type: to_float_type(keys)),
        Nx.u8(1),
        indices,
        keys,
        flags
      ])

    {indices, keys, flags} =
      add_neighbor_vectorized({indices, keys, flags}, iota, zero_node, new_flag, num_nodes)

    {Nx.revectorize(indices, indices_vectorized_axes, target_shape: {num_heaps, num_nodes}),
     Nx.revectorize(keys, keys_vectorized_axes, target_shape: {num_heaps, num_nodes}),
     Nx.revectorize(flags, flags_vectorized_axes, target_shape: {num_heaps, num_nodes})}
  end

  # Updates the nearest neighbor graph using leaves constructed from random projection trees.

  # This function updates the nearest neighbor graph by incorporating the
  # information from the leaves constructed from random projection trees.
  defnp update_by_leaves(data, curr_graph, leaves) do
    num_leaves = Nx.axis_size(leaves, 0)
    leaf_size = Nx.axis_size(leaves, 1)

    updates_index = Nx.s64(0)
    updates_indices = Nx.broadcast(Nx.s64(0), {num_leaves, 2})
    updates_dist = Nx.broadcast(Nx.tensor(0.0, type: to_float_type(data)), {num_leaves})

    {_indices, keys, _flags} = curr_graph

    while {{updates_indices, updates_dist, updates_index}, {i = Nx.s64(0), keys, data, leaves}},
          i < num_leaves do
      {{updates_indices, updates_dist, updates_index}, _} =
        while {{updates_indices, updates_dist, updates_index},
               {i, j = Nx.s64(0), keys, data, leaves, stop = Nx.u8(0)}},
              j < leaf_size and not stop do
          index0 = leaves[[i, j]]

          if index0 != Nx.s64(-1) do
            {{updates_indices, updates_dist, updates_index}, _} =
              while {{updates_indices, updates_dist, updates_index},
                     {i, j, k = j + 1, keys, index0, data, leaves, stop_inner = Nx.u8(0)}},
                    k < leaf_size and not stop_inner do
                index1 = leaves[[i, k]]

                if index1 != Nx.s64(-1) do
                  d = Distance.squared_euclidean(data[index0], data[index1])

                  {updates_indices, updates_dist, updates_index} =
                    if d < keys[[index0, 0]] or d < keys[[index1, 0]] do
                      updates_indices =
                        Nx.put_slice(
                          updates_indices,
                          [updates_index, 0],
                          Nx.new_axis(
                            Nx.concatenate([Nx.new_axis(index0, 0), Nx.new_axis(index1, 0)]),
                            0
                          )
                        )

                      updates_dist =
                        Nx.indexed_put(updates_dist, Nx.new_axis(updates_index, 0), d)

                      {updates_indices, updates_dist, updates_index + 1}
                    else
                      {updates_indices, updates_dist, updates_index}
                    end

                  {{updates_indices, updates_dist, updates_index},
                   {i, j, k + 1, keys, index0, data, leaves, Nx.u8(0)}}
                else
                  {{updates_indices, updates_dist, updates_index},
                   {i, j, k + 1, keys, index0, data, leaves, Nx.u8(1)}}
                end
              end

            {{updates_indices, updates_dist, updates_index},
             {i, j + 1, keys, data, leaves, Nx.u8(0)}}
          else
            {{updates_indices, updates_dist, updates_index},
             {i, j + 1, keys, data, leaves, Nx.u8(1)}}
          end
        end

      {{updates_indices, updates_dist, updates_index}, {i + 1, keys, data, leaves}}
    end

    {curr_graph, _} =
      while {curr_graph, {i = Nx.s64(0), updates_index, updates_indices, updates_dist}},
            i < updates_index do
        index0 = updates_indices[[i, 0]]
        index1 = updates_indices[[i, 1]]
        d = updates_dist[i]
        curr_graph = add_neighbor(curr_graph, index0, index1, d, Nx.s8(1))
        {curr_graph, {i + 1, updates_index, updates_indices, updates_dist}}
      end

    curr_graph
  end

  # Builds a heap of candidate neighbors for nearest neighbor descent.

  # For each vertex, the candidate neighbors include any current neighbors and
  # any vertices that have the vertex as one of their nearest neighbors.
  defnp sample_candidate(curr_graph, new_candidates, old_candidates, rng_key) do
    {indices, keys, flags} = curr_graph
    {num_heaps, num_nodes} = Nx.shape(indices)

    {{{indices, keys, flags}, new_candidates, old_candidates, rng_key}, _} =
      while {{{indices, keys, flags}, new_candidates, old_candidates, rng_key}, i = Nx.s64(0)},
            i < num_heaps do
        {{{indices, keys, flags}, new_candidates, old_candidates, rng_key}, _} =
          while {{{indices, keys, flags}, new_candidates, old_candidates, rng_key},
                 {j = Nx.s64(0), i}},
                j < num_nodes do
            index1 = indices[[i, j]]
            flag = flags[[i, j]]

            if index1 == Nx.s64(-1) do
              {{{indices, keys, flags}, new_candidates, old_candidates, rng_key}, {j + 1, i}}
            else
              {priority, new_rng_key} =
                Nx.Random.randint(rng_key, 1, Nx.Constants.max(:s64), type: :s64)

              if flag == Nx.u8(1) do
                new_candidates = add_neighbor(new_candidates, i, index1, priority, Nx.s8(-1))
                new_candidates = add_neighbor(new_candidates, index1, i, priority, Nx.s8(-1))

                {{{indices, keys, flags}, new_candidates, old_candidates, new_rng_key},
                 {j + 1, i}}
              else
                old_candidates = add_neighbor(old_candidates, i, index1, priority, Nx.s8(-1))
                old_candidates = add_neighbor(old_candidates, index1, i, priority, Nx.s8(-1))

                {{{indices, keys, flags}, new_candidates, old_candidates, new_rng_key},
                 {j + 1, i}}
              end
            end
          end

        {{{indices, keys, flags}, new_candidates, old_candidates, rng_key}, i + 1}
      end

    {new_candidates_indices, _new_candidates_keys, _new_candidates_flags} = new_candidates
    new_candidates_num_nodes = Nx.axis_size(new_candidates_indices, 1)

    {{{indices, keys, flags}, _new_candidates_indices}, _} =
      while {{{indices, keys, flags}, new_candidates_indices}, i = Nx.s64(0)}, i < num_heaps do
        {{{indices, keys, flags}, new_candidates_indices}, _} =
          while {{{indices, keys, flags}, new_candidates_indices}, {j = Nx.s64(0), i}},
                j < num_nodes do
            index1 = indices[[i, j]]
            stop = Nx.u8(0)

            {{{indices, keys, flags}, new_candidates_indices}, _} =
              while {{{indices, keys, flags}, new_candidates_indices},
                     {stop, index1, k = Nx.s64(0), i, j}},
                    k < new_candidates_num_nodes and not stop do
                {{indices, keys, flags}, stop} =
                  if new_candidates_indices[[i, k]] == index1 do
                    flags =
                      Nx.indexed_put(
                        flags,
                        Nx.concatenate([Nx.new_axis(i, 0), Nx.new_axis(j, 0)]),
                        Nx.s8(0)
                      )

                    {{indices, keys, flags}, Nx.u8(1)}
                  else
                    {{indices, keys, flags}, Nx.u8(0)}
                  end

                {{{indices, keys, flags}, new_candidates_indices}, {stop, index1, k + 1, i, j}}
              end

            {{{indices, keys, flags}, new_candidates_indices}, {j + 1, i}}
          end

        {{{indices, keys, flags}, new_candidates_indices}, i + 1}
      end

    {{indices, keys, flags}, new_candidates, old_candidates, rng_key}
  end

  # This function generates potential nearest neighbor updates, which are
  # objects containing two identifiers that identify nodes and their
  # corresponding distance.
  defnp generate_graph_updates(
          data,
          curr_graph,
          new_candidate_neighbors,
          old_candidate_neighbors
        ) do
    {_indices, keys, _flags} = curr_graph

    {old_candidates_indices, _old_candidates_keys, _old_candidates_flags} =
      old_candidate_neighbors

    {new_candidates_indices, _new_candidates_keys, _new_candidates_flags} =
      new_candidate_neighbors

    {size_new, new_candidates_num_nodes} = Nx.shape(new_candidates_indices)
    old_candidates_num_nodes = Nx.axis_size(old_candidates_indices, 1)
    num_samples = Nx.axis_size(data, 0)
    update_index = Nx.s64(0)

    # Normally there would be a stack of that will dynamically grow
    # so we need to preallocate it with a fixed size
    expand_factor = 50
    updates_indices = Nx.broadcast(Nx.s64(0), {expand_factor * num_samples, 2})

    updates_dist =
      Nx.broadcast(Nx.tensor(0.0, type: to_float_type(data)), {expand_factor * num_samples})

    {{_keys, _new_candidates_indices, _old_candidates_indices, updates_indices, updates_dist,
      update_index},
     _} =
      while {{keys, new_candidates_indices, old_candidates_indices, updates_indices, updates_dist,
              update_index}, {i = Nx.s64(0), data}},
            i < size_new do
        {{_, _, _, updates_indices, updates_dist, update_index}, _} =
          while {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                  updates_dist, update_index}, {j = Nx.s64(0), data, i}},
                j < new_candidates_num_nodes do
            index0 = new_candidates_indices[[i, j]]

            if index0 == Nx.s64(-1) do
              {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                updates_dist, update_index}, {j + 1, data, i}}
            else
              {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                updates_dist, update_index},
               _} =
                while {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                        updates_dist, update_index}, {k = j + 1, data, i, j, index0}},
                      k < new_candidates_num_nodes do
                  index1 = new_candidates_indices[[i, k]]

                  if index1 == Nx.s64(-1) do
                    {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                      updates_dist, update_index}, {k + 1, data, i, j, index0}}
                  else
                    d = Distance.squared_euclidean(data[index0], data[index1])

                    {updates_indices, updates_dist, update_index} =
                      if d < keys[[index0, 0]] or d < keys[[index1, 0]] do
                        updates_indices =
                          Nx.put_slice(
                            updates_indices,
                            [update_index, 0],
                            Nx.new_axis(
                              Nx.concatenate([Nx.new_axis(index0, 0), Nx.new_axis(index1, 0)]),
                              0
                            )
                          )

                        updates_dist =
                          Nx.indexed_put(updates_dist, Nx.new_axis(update_index, 0), d)

                        {updates_indices, updates_dist, update_index + 1}
                      else
                        {updates_indices, updates_dist, update_index}
                      end

                    {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                      updates_dist, update_index}, {k + 1, data, i, j, index0}}
                  end
                end

              {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                updates_dist, update_index},
               _} =
                while {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                        updates_dist, update_index}, {k = Nx.s64(0), data, i, index0}},
                      k < old_candidates_num_nodes do
                  index1 = old_candidates_indices[[i, k]]

                  if index1 == Nx.s64(-1) do
                    {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                      updates_dist, update_index}, {k + 1, data, i, index0}}
                  else
                    d = Distance.squared_euclidean(data[index0], data[index1])

                    {updates_indices, updates_dist, update_index} =
                      if d < keys[[index0, 0]] or d < keys[[index1, 0]] do
                        updates_indices =
                          Nx.put_slice(
                            updates_indices,
                            [update_index, 0],
                            Nx.new_axis(
                              Nx.concatenate([Nx.new_axis(index0, 0), Nx.new_axis(index1, 0)]),
                              0
                            )
                          )

                        updates_dist =
                          Nx.indexed_put(updates_dist, Nx.new_axis(update_index, 0), d)

                        {updates_indices, updates_dist, update_index + 1}
                      else
                        {updates_indices, updates_dist, update_index}
                      end

                    {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                      updates_dist, update_index}, {k + 1, data, i, index0}}
                  end
                end

              {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
                updates_dist, update_index}, {j + 1, data, i}}
            end
          end

        {{keys, new_candidates_indices, old_candidates_indices, updates_indices, updates_dist,
          update_index}, {i + 1, data}}
      end

    {updates_indices, updates_dist, update_index}
  end

  # Applies graph updates to the current nearest neighbor graph.
  defnp apply_updates(curr_graph, updates_indices, updates_dist, update_index) do
    {curr_graph, _} =
      while {curr_graph, {i = Nx.s64(0), update_index, updates_indices, updates_dist}},
            i < update_index do
        index0 = updates_indices[[i, 0]]
        index1 = updates_indices[[i, 1]]
        d = updates_dist[i]

        curr_graph = add_neighbor(curr_graph, index0, index1, d, Nx.s8(1))
        curr_graph = add_neighbor(curr_graph, index1, index0, d, Nx.s8(1))
        {curr_graph, {i + 1, update_index, updates_indices, updates_dist}}
      end

    curr_graph
  end

  # This function applies the NN-descent algorithm to construct an approximate
  # nearest neighbor graph. It iteratively refines the graph by exploring
  # neighbor candidates and updating the graph connections based on the
  # distances between nodes. The algorithm aims to find a graph that represents
  # the nearest neighbor relationships in the data.
  defnp nn_descent(data, curr_graph, key, opts \\ []) do
    max_iters = opts[:max_iterations]
    tol = opts[:tol]
    max_candidates = opts[:max_candidates]
    num_neighbors = opts[:num_neighbors]
    num_samples = Nx.axis_size(data, 0)
    stop = Nx.u8(0)

    {curr_graph, _} =
      while {curr_graph, {i = Nx.u64(0), stop, key, data}},
            i < max_iters and not stop do
        new_candidates =
          {Nx.broadcast(Nx.s64(-1), {num_samples, max_candidates}),
           Nx.broadcast(Nx.Constants.max_finite(:s64), {num_samples, max_candidates}),
           Nx.broadcast(Nx.s8(0), {num_samples, max_candidates})}

        old_candidates =
          {Nx.broadcast(Nx.s64(-1), {num_samples, max_candidates}),
           Nx.broadcast(Nx.Constants.max_finite(:s64), {num_samples, max_candidates}),
           Nx.broadcast(Nx.s8(0), {num_samples, max_candidates})}

        {curr_graph, new_candidates, old_candidates, key} =
          sample_candidate(curr_graph, new_candidates, old_candidates, key)

        {updates_indices, updates_dist, update_index} =
          generate_graph_updates(data, curr_graph, new_candidates, old_candidates)

        curr_graph = apply_updates(curr_graph, updates_indices, updates_dist, update_index)

        stop =
          if update_index < tol * num_samples * num_neighbors do
            Nx.u8(1)
          else
            Nx.u8(0)
          end

        {curr_graph, {i + 1, stop, key, data}}
      end

    {indices, dists, _flags} = add_zero_nodes(curr_graph)
    ord = Nx.argsort(dists, axis: 1)

    %__MODULE__{
      nearest_neighbors: Nx.take_along_axis(indices, ord, axis: 1),
      distances: Nx.take_along_axis(dists, ord, axis: 1)
    }
  end

  # Retrieve the leaves from random projection forests.
  defnp get_leaves_from_forest(forest) do
    leaf_size = forest.leaf_size
    {num_trees, num_indices} = Nx.shape(forest.indices)
    to_concat = rem(num_indices, leaf_size)

    leaves =
      if to_concat != Nx.s64(0) do
        Nx.concatenate(
          [
            forest.indices,
            Nx.broadcast(Nx.s64(-1), {num_trees, leaf_size - to_concat})
          ],
          axis: 1
        )
      else
        forest.indices
      end

    Nx.reshape(leaves, {:auto, leaf_size})
  end

  # Procedure that adds neighbor to the graph according to the provided distance (key)
  # and flag.
  defnp add_neighbor(curr_graph, index0, index1, key, flag) do
    {indices, keys, flags} = curr_graph
    num_nodes = Nx.axis_size(indices, 1)

    stop =
      if key >= keys[[index0, 0]] do
        Nx.u8(1)
      else
        Nx.u8(0)
      end

    stop = stop or Nx.any(indices[index0] == index1)

    if stop do
      curr_graph
    else
      {{indices, keys, flags, curr, _}, _} =
        while {{indices, keys, flags, curr = Nx.s64(0), swap = Nx.s64(0)},
               {index0, stop, key, flag}},
              not stop do
          left_child = 2 * curr + 1
          right_child = left_child + 1

          {swap, stop} =
            cond do
              left_child >= num_nodes ->
                {swap, Nx.u8(1)}

              right_child >= num_nodes and keys[[index0, left_child]] > key ->
                {left_child, stop}

              right_child >= num_nodes and keys[[index0, left_child]] <= key ->
                {swap, Nx.u8(1)}

              keys[[index0, left_child]] >= keys[[index0, right_child]] and
                  keys[[index0, left_child]] > key ->
                {left_child, stop}

              keys[[index0, left_child]] >= keys[[index0, right_child]] and
                  keys[[index0, left_child]] <= key ->
                {swap, Nx.u8(1)}

              keys[[index0, right_child]] > key ->
                {right_child, stop}

              keys[[index0, right_child]] <= key ->
                {swap, Nx.u8(1)}

              true ->
                {swap, Nx.u8(1)}
            end

          index0_ext = Nx.new_axis(index0, 0)
          curr_ext = Nx.new_axis(curr, 0)

          indices =
            Nx.indexed_put(
              indices,
              Nx.concatenate([index0_ext, curr_ext]),
              indices[[index0, swap]]
            )

          keys =
            Nx.indexed_put(keys, Nx.concatenate([index0_ext, curr_ext]), keys[[index0, swap]])

          flags =
            if flag != Nx.s8(-1),
              do:
                Nx.indexed_put(
                  flags,
                  Nx.concatenate([index0_ext, curr_ext]),
                  flags[[index0, swap]]
                ),
              else: flags

          curr = swap
          {{indices, keys, flags, curr, swap}, {index0, stop, key, flag}}
        end

      index0_ext = Nx.new_axis(index0, 0)
      curr_ext = Nx.new_axis(curr, 0)
      indices = Nx.indexed_put(indices, Nx.concatenate([index0_ext, curr_ext]), index1)
      keys = Nx.indexed_put(keys, Nx.concatenate([index0_ext, curr_ext]), key)

      flags =
        if flag != Nx.s8(-1),
          do: Nx.indexed_put(flags, Nx.concatenate([index0_ext, curr_ext]), flag),
          else: flags

      {indices, keys, flags}
    end
  end

  defnp add_neighbor_vectorized(curr_graph, index1, key, flag, num_nodes) do
    {indices, keys, flags} = curr_graph

    stop =
      if key >= keys[[0]] do
        Nx.u8(1)
      else
        Nx.u8(0)
      end

    stop = stop or Nx.any(indices == index1)

    if stop do
      curr_graph
    else
      {{indices, keys, flags, curr, _}, _} =
        while {{indices, keys, flags, curr = Nx.s64(0), swap = Nx.s64(0)}, {stop, key, flag}},
              not stop do
          left_child = 2 * curr + 1
          right_child = left_child + 1

          {swap, stop} =
            cond do
              left_child >= num_nodes ->
                {swap, Nx.u8(1)}

              right_child >= num_nodes and keys[[left_child]] > key ->
                {left_child, stop}

              right_child >= num_nodes and keys[[left_child]] <= key ->
                {swap, Nx.u8(1)}

              keys[[left_child]] >= keys[[right_child]] and
                  keys[[left_child]] > key ->
                {left_child, stop}

              keys[[left_child]] >= keys[[right_child]] and
                  keys[[left_child]] <= key ->
                {swap, Nx.u8(1)}

              keys[[right_child]] > key ->
                {right_child, stop}

              keys[[right_child]] <= key ->
                {swap, Nx.u8(1)}

              true ->
                {swap, Nx.u8(1)}
            end

          curr_ext = Nx.new_axis(curr, 0)

          indices =
            Nx.indexed_put(
              indices,
              Nx.concatenate([curr_ext]),
              indices[[swap]]
            )

          keys =
            Nx.indexed_put(keys, Nx.concatenate([curr_ext]), keys[[swap]])

          flags =
            if flag != Nx.s8(-1),
              do:
                Nx.indexed_put(
                  flags,
                  Nx.concatenate([curr_ext]),
                  flags[[swap]]
                ),
              else: flags

          curr = swap
          {{indices, keys, flags, curr, swap}, {stop, key, flag}}
        end

      curr_ext = Nx.new_axis(curr, 0)
      indices = Nx.indexed_put(indices, Nx.concatenate([curr_ext]), index1)
      keys = Nx.indexed_put(keys, Nx.concatenate([curr_ext]), key)

      flags =
        if flag != Nx.s8(-1),
          do: Nx.indexed_put(flags, Nx.concatenate([curr_ext]), flag),
          else: flags

      {indices, keys, flags}
    end
  end
end
