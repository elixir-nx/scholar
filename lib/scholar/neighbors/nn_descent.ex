defmodule Scholar.Neighbors.NNDescent do
  @moduledoc """
  Nearest Neighbors Descent (NND) is an algorithm that calculates Approximated Nearest Neighbors (ANN)
  for a given set of points[1].

  It is implemented using Random Projection Forest[2].

  ## References

    [1] [Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures](https://www.cs.princeton.edu/cass/papers/www11.pdf).
    [2] [Randomized partition trees for nearest neighbor search](https://cseweb.ucsd.edu/~dasgupta/papers/exactnn-algo.pdf)
  """

  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Metrics.Distance
  alias Scholar.Neighbors.RandomProjectionForest

  @derive {Nx.Container,
           containers: [:nearest_neighbors, :distances, :forest, :train_data],
           keep: [:is_forest_computed?]}
  defstruct [:nearest_neighbors, :distances, :forest, :train_data, :is_forest_computed?]

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
      default: 1.0e-3,
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

  query_opts = [
    pruning_probability: [
      type: :float,
      default: 0.8,
      doc: """
      The probability of pruning a dimension in the random projection tree.
      """
    ],
    prunning_factor: [
      type: :float,
      default: 1.5,
      doc: """
      # TODO check this description
      The factor by which the number of points in a leaf should be smaller than
      the number of points in the root node.
      """
    ],
    eps: [
      type: :float,
      default: 1.0e-3,
      doc: """
      The minimum value that distinguish between two different points.
      """
    ],
    num_neighbors: [
      type: :pos_integer,
      required: true,
      doc: "The number of neighbors to use in queries"
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Determines random number generation for centroid initialization.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)
  @query_opts_schema NimbleOptions.new!(query_opts)

  @doc """
  Calculates the approximate nearest neighbors for a given set of points. It
  returns a struct containing two tensors:

  * `nearest_neighbors` - the indices of the nearest neighbors
  * `distances` - the distances to the nearest neighbors

  ## Examples

      iex> data = Nx.iota({10, 5})
      iex> key = Nx.Random.key(12)
      iex> Scholar.Neighbors.NNDescent.fit(data, num_neighbors: 3, key: key)
  """
  deftransform fit(tensor, opts \\ []) do
    opts =
      if Keyword.has_key?(opts, :max_candidates) do
        opts
      else
        Keyword.put(opts, :max_candidates, min(60, opts[:num_neighbors]))
      end

    opts = NimbleOptions.validate!(opts, @opts_schema)

    opts =
      Keyword.put(opts, :num_trees, 5 + Kernel.round(:math.pow(Nx.axis_size(tensor, 0), 0.25)))

    num_samples = Nx.axis_size(tensor, 0)

    if opts[:max_candidates] > num_samples do
      raise ArgumentError,
            """
            expected max_candidates to be less than or equal to the number of samples, \
            got max_candidates: #{opts[:max_candidates]} and number of samples: \
            #{num_samples}
            """
    end

    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)

    opts = Keyword.put(opts, :max_candidates, min(opts[:max_candidates], opts[:num_neighbors]))

    nn_descent(tensor, key, opts)
  end

  # Initializes the graph. It returns a tuple of three tensors:

  # * `indices` - the indices of the neighbors
  # * `keys` - the distances to the neighbors
  # * `flags` - flags indicating whether the neighbor is a new candidate or an old candidate
  defnp initialize_graph(tensor, opts) do
    num_neighbors = opts[:num_neighbors]
    num_samples = Nx.axis_size(tensor, 0)

    {Nx.broadcast(Nx.s64(-1), {num_samples, num_neighbors}),
     Nx.broadcast(Nx.Constants.infinity(to_float_type(tensor)), {num_samples, num_neighbors}),
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

  # Updates the nearest neighbor graph using random projection forest.

  # This function updates the nearest neighbor graph by incorporating the
  # information from the random projection forest
  defnp update_by_forest(data, curr_graph, forest) do
    {indices, distances} = Scholar.Neighbors.RandomProjectionForest.predict(forest, data)

    {curr_graph, _} =
      while {curr_graph, {i = Nx.s64(0), indices, distances}},
            i < Nx.axis_size(indices, 0) do
        {curr_graph, _} =
          while {curr_graph, {i, j = 0, indices, distances}}, j < Nx.axis_size(data, 1) do
            curr_graph = add_neighbor(curr_graph, i, indices[i][j], distances[i][j], Nx.s8(1))
            {curr_graph, {i, j + 1, indices, distances}}
          end

        {curr_graph, {i + 1, indices, distances}}
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
  defnp nn_descent(data, key, opts \\ []) do
    curr_graph = initialize_graph(data, opts)

    {curr_graph, forest} =
      if opts[:tree_init?] do
        forest =
          RandomProjectionForest.fit(data,
            key: key,
            num_trees: opts[:num_trees],
            min_leaf_size: min(div(Nx.axis_size(data, 0), 2), max(opts[:num_neighbors], 10)),
            num_neighbors: opts[:num_neighbors]
          )

        {update_by_forest(data, curr_graph, forest), forest}
      else
        {curr_graph, Nx.tensor(:nan)}
      end

    {curr_graph, key} = init_random(data, curr_graph, key, opts)

    max_iters = opts[:max_iterations]
    tol = opts[:tol]
    max_candidates = opts[:max_candidates]
    num_neighbors = min(opts[:num_neighbors], Nx.axis_size(data, 0))
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
    ord = Nx.argsort(dists, axis: 1, stable: true)

    %__MODULE__{
      nearest_neighbors: Nx.take_along_axis(indices, ord, axis: 1),
      distances: Nx.take_along_axis(dists, ord, axis: 1),
      forest: forest,
      train_data: data,
      is_forest_computed?: opts[:tree_init?]
    }
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

  defnp prune_long_edges(data, indices, keys, rng_key, opts) do
    pruning_probability = opts[:pruning_probability]
    {num_heaps, num_nodes} = Nx.shape(indices)

    new_indices_init = Nx.broadcast(Nx.s64(-1), {num_nodes})
    new_keys_init = Nx.broadcast(Nx.Constants.max_finite(to_float_type(data)), {num_nodes})

    {{indices, keys}, _} =
      while {{indices, keys}, {i = Nx.s64(0), data, new_indices_init, new_keys_init, rng_key}},
            i < num_heaps do
        new_indices_size = Nx.s64(0)
        new_indices = new_indices_init
        new_keys = new_keys_init

        # First element -> node itself so we prune it.
        {{new_indices, new_keys, _new_indices_size}, _} =
          while {{new_indices, new_keys, new_indices_size},
                 {j = Nx.s64(1), i, data, indices, keys, rng_key}},
                j < num_nodes do
            index = indices[[i, j]]
            key = keys[[i, j]]

            if index == Nx.s64(-1) do
              {{new_indices, new_keys, new_indices_size},
               {j + 1, i, data, indices, keys, rng_key}}
            else
              add_node? = Nx.u8(1)
              stop = Nx.u8(0)

              {add_node?, _} =
                while {add_node?,
                       {k = Nx.s64(0), i, index, key, data, stop, new_indices, new_keys,
                        new_indices_size, rng_key}},
                      k < new_indices_size and not stop do
                  new_index = new_indices[[k]]
                  new_key = new_keys[[k]]

                  d = Distance.squared_euclidean(data[index], data[new_index])

                  {add_node?, stop, rng_key} =
                    if new_key > opts[:eps] and d < key do
                      {temp, rng_key} = Nx.Random.uniform(rng_key, type: :f32)

                      if temp < pruning_probability do
                        {Nx.u8(0), Nx.u8(1), rng_key}
                      else
                        {add_node?, stop, rng_key}
                      end
                    else
                      {add_node?, stop, rng_key}
                    end

                  {add_node?,
                   {k + 1, i, index, key, data, stop, new_indices, new_keys, new_indices_size,
                    rng_key}}
                end

              {new_indices, new_keys, new_indices_size} =
                if add_node? do
                  new_indices =
                    Nx.indexed_put(new_indices, Nx.new_axis(new_indices_size, 0), index)

                  new_keys = Nx.indexed_put(new_keys, Nx.new_axis(new_indices_size, 0), key)
                  {new_indices, new_keys, new_indices_size + 1}
                else
                  {new_indices, new_keys, new_indices_size}
                end

              {{new_indices, new_keys, new_indices_size},
               {j + 1, i, data, indices, keys, rng_key}}
            end
          end

        indices = Nx.put_slice(indices, [i, 0], Nx.new_axis(new_indices, 0))
        keys = Nx.put_slice(keys, [i, 0], Nx.new_axis(new_keys, 0))

        {{indices, keys}, {i + 1, data, new_indices_init, new_keys_init, rng_key}}
      end

    {indices, keys}
  end

  # prepare all needed data for query

  defnp prepare(data, indices, keys, key, opts) do
    {num_heaps, num_nodes} = Nx.shape(indices)

    {indices, keys} = prune_long_edges(data, indices, keys, key, opts)

    search_graph = initialize_graph(data, opts)

    {search_graph, _} =
      while {search_graph, {indices, keys, i = 0}}, i < num_heaps do
        {{search_graph, indices, keys, i}, _} =
          while {{search_graph, indices, keys, i}, j = 0}, j < num_nodes do
            index = indices[[i, j]]

            search_graph =
              if index != Nx.s64(-1) do
                d = keys[[i, j]]
                search_graph = add_neighbor(search_graph, i, index, d, Nx.s8(-1))
                add_neighbor(search_graph, index, i, d, Nx.s8(-1))
              else
                search_graph
              end

            {{search_graph, indices, keys, i}, j + 1}
          end

        {search_graph, {indices, keys, i + 1}}
      end

    {search_indices, search_distances, search_flags} = search_graph
    ord = Nx.argsort(search_distances, axis: 1, stable: true)

    search_indices = Nx.take_along_axis(search_indices, ord, axis: 1)
    search_distances = Nx.take_along_axis(search_distances, ord, axis: 1)

    {search_indices, search_distances, search_flags}
  end

  @doc """
  Query the training data for the nearest neighbors of the query data.

  Use this function if you query for the first time on a training data.
  It will compute search graph that can be reused in the future queries.

  ## Examples

      iex> data = Nx.iota({100, 5})
      iex> query_data = Nx.tensor([[1,7,32,6,2], [1,4,5,2,5], [1,3,67,12,4]])
      iex> key = Nx.Random.key(12)
      iex> nn = Scholar.Neighbors.NNDescent.fit(data, num_neighbors: 6)
      iex> Scholar.Neighbors.NNDescent.query(nn, query_data, num_neighbors: 5, key: key)
  """
  deftransform query(
                 %__MODULE__{
                   nearest_neighbors: nearest_neighbors,
                   distances: distances,
                   forest: forest,
                   is_forest_computed?: is_forest_computed?,
                   train_data: train_data
                 } = nn,
                 query_data,
                 opts
               ) do
    opts = NimbleOptions.validate!(opts, @query_opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    search_graph = prepare(train_data, nearest_neighbors, distances, key, opts)

    forest =
      if is_forest_computed? do
        forest
      else
        RandomProjectionForest.fit(train_data, opts)
      end

    query_n(
      nn,
      query_data,
      train_data,
      search_graph,
      forest,
      key,
      opts
    )
  end

  @doc """
  Query the training data for the nearest neighbors of the query data.

  Use this function if you have already queried training data.
  It will compute search graph that can be reused in the future queries.

  ## Examples

      iex> data = Nx.iota({100, 5})
      iex> query_data = Nx.tensor([[1,7,32,6,2], [1,4,5,2,5], [1,3,67,12,4]])
      iex> key = Nx.Random.key(12)
      iex> nn = Scholar.Neighbors.NNDescent.fit(data, num_neighbors: 6)
      iex> {indices, distances, search_graph} = Scholar.Neighbors.NNDescent.query(nn, query_data, num_neighbors: 5, key: key)
      iex> query_data2 = Nx.tensor([[2,7,32,6,2], [1,4,6,2,5], [1,3,67,12,3]])
      iex> Scholar.Neighbors.NNDescent.query(nn, query_data2, search_graph, num_neighbors: 5, key: key)
  """
  deftransform query(
                 %__MODULE__{
                   forest: forest,
                   is_forest_computed?: is_forest_computed?,
                   train_data: train_data
                 } = nn,
                 query_data,
                 search_graph,
                 opts
               ) do
    opts = NimbleOptions.validate!(opts, @query_opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)

    forest =
      if is_forest_computed? do
        forest
      else
        RandomProjectionForest.fit(train_data, opts)
      end

    query_n(
      nn,
      query_data,
      train_data,
      search_graph,
      forest,
      key,
      opts
    )
  end

  defn query_n(
         %__MODULE__{nearest_neighbors: nearest_neighbors},
         query_data,
         train_data,
         search_graph,
         forest,
         rng_key,
         opts
       ) do
    train_data_size = Nx.axis_size(train_data, 0)
    {query_indices, query_keys, query_flags} = initialize_graph(query_data, opts)

    {num_heaps, num_nodes} = Nx.shape(query_indices)

    {initial_candidates, initial_distances} = RandomProjectionForest.predict(forest, query_data)

    {{query_indices, query_keys, _query_flags}, _query_data, search_graph, _initial_candidates,
     _initial_distances, _train_data, _rng_key,
     _i} =
      while {{query_indices, query_keys, query_flags}, query_data, search_graph,
             initial_candidates, initial_distances, train_data, rng_key, i = 0},
            i < num_heaps do
        visited = Nx.broadcast(Nx.u8(0), {train_data_size})
        expand_factor = 50
        search_candidates_indices = Nx.broadcast(Nx.s64(-1), {num_nodes * expand_factor})

        search_candidates_distances =
          Nx.broadcast(
            Nx.Constants.max_finite(to_float_type(query_data)),
            {num_nodes * expand_factor}
          )

        search_ptr = Nx.s64(0)

        {{query_indices, query_keys, query_flags},
         {visited, search_candidates_indices, search_candidates_distances, search_ptr},
         {query_data, search_graph, initial_candidates, initial_distances, train_data, i, _stop,
          j}} =
          while {{query_indices, query_keys, query_flags},
                 {visited, search_candidates_indices, search_candidates_distances, search_ptr},
                 {query_data, search_graph, initial_candidates, initial_distances, train_data, i,
                  stop = Nx.u8(0), j = 0}},
                j < Nx.axis_size(nearest_neighbors, 1) and not stop do
            if initial_candidates[i][j] != Nx.s64(-1) do
              d = Distance.squared_euclidean(train_data[initial_candidates[i][j]], query_data[i])

              visited =
                Nx.indexed_put(visited, Nx.new_axis(initial_candidates[i][j], 0), Nx.u8(1))

              {query_indices, query_keys, query_flags} =
                add_neighbor(
                  {query_indices, query_keys, query_flags},
                  i,
                  initial_candidates[i][j],
                  d,
                  Nx.s8(-1)
                )

              search_candidates_indices =
                Nx.indexed_put(
                  search_candidates_indices,
                  Nx.new_axis(search_ptr, 0),
                  initial_candidates[i][j]
                )

              search_candidates_distances =
                Nx.indexed_put(search_candidates_distances, Nx.new_axis(search_ptr, 0), d)

              search_ptr = search_ptr + 1

              {{query_indices, query_keys, query_flags},
               {visited, search_candidates_indices, search_candidates_distances, search_ptr},
               {query_data, search_graph, initial_candidates, initial_distances, train_data, i,
                Nx.u8(0), j + 1}}
            else
              {{query_indices, query_keys, query_flags},
               {visited, search_candidates_indices, search_candidates_distances, search_ptr},
               {query_data, search_graph, initial_candidates, initial_distances, train_data, i,
                Nx.u8(1), j}}
            end
          end

        {{query_indices, query_keys, query_flags},
         {visited, search_candidates_indices, search_candidates_distances, search_ptr},
         _} =
          while {{query_indices, query_keys, query_flags},
                 {visited, search_candidates_indices, search_candidates_distances, search_ptr},
                 {query_data, search_graph, initial_candidates, initial_distances, train_data,
                  rng_key, i, stop = Nx.u8(0), k = j}},
                k < Nx.axis_size(nearest_neighbors, 1) and not stop do
            {index, rng_key} = Nx.Random.randint(rng_key, 0, Nx.axis_size(train_data, 0))

            if not visited[index] do
              d = Distance.squared_euclidean(train_data[index], query_data[i])
              visited = Nx.indexed_put(visited, Nx.new_axis(index, 0), Nx.u8(1))

              {query_indices, query_keys, query_flags} =
                add_neighbor(
                  {query_indices, query_keys, query_flags},
                  i,
                  index,
                  d,
                  Nx.s8(-1)
                )

              search_candidates_indices =
                Nx.indexed_put(
                  search_candidates_indices,
                  Nx.new_axis(search_ptr, 0),
                  index
                )

              search_candidates_distances =
                Nx.indexed_put(search_candidates_distances, Nx.new_axis(search_ptr, 0), d)

              search_ptr = search_ptr + 1

              {{query_indices, query_keys, query_flags},
               {visited, search_candidates_indices, search_candidates_distances, search_ptr},
               {query_data, search_graph, initial_candidates, initial_distances, train_data,
                rng_key, i, Nx.u8(0), k + 1}}
            else
              {{query_indices, query_keys, query_flags},
               {visited, search_candidates_indices, search_candidates_distances, search_ptr},
               {query_data, search_graph, initial_candidates, initial_distances, train_data,
                rng_key, i, Nx.u8(1), k}}
            end
          end

        ord = Nx.argsort(search_candidates_distances, stable: true)
        search_candidates_indices = Nx.take_along_axis(search_candidates_indices, ord)
        search_candidates_distances = Nx.take_along_axis(search_candidates_distances, ord)

        index = Nx.s64(0)

        dist_bound =
          (Nx.tensor(1.0, type: to_float_type(query_data)) + opts[:eps]) * query_keys[[i, 0]]

        {search_indices, search_keys, search_flags} = search_graph

        while {{query_indices, query_keys, query_flags},
               {visited, search_candidates_indices, search_candidates_distances, search_ptr},
               query_data, {search_indices, search_keys, search_flags}, initial_candidates,
               initial_distances, train_data, i, dist_bound, stop, index},
              index < search_ptr and search_candidates_distances[[index]] and
                not stop < dist_bound do
          candidate = search_candidates_indices[[index]]

          {{query_indices, query_keys, query_flags},
           {visited, search_candidates_indices, search_candidates_distances, search_ptr},
           query_data, _, initial_candidates, initial_distances, train_data, i, dist_bound, stop,
           index, _,
           _} =
            while {{query_indices, query_keys, query_flags},
                   {visited, search_candidates_indices, search_candidates_distances, search_ptr},
                   query_data, {search_indices, search_keys, search_flags}, initial_candidates,
                   initial_distances, train_data, i, dist_bound, stop, index, candidate, k = 0},
                  k < num_nodes do
              j = search_indices[[candidate, k]]

              if j == Nx.s64(-1) do
                {{query_indices, query_keys, query_flags},
                 {visited, search_candidates_indices, search_candidates_distances, search_ptr},
                 query_data, search_graph, initial_candidates, initial_distances, train_data, i,
                 dist_bound, Nx.u8(1), index, candidate, k}
              else
                if visited[j] do
                  {{query_indices, query_keys, query_flags},
                   {visited, search_candidates_indices, search_candidates_distances, search_ptr},
                   query_data, search_graph, initial_candidates, initial_distances, train_data, i,
                   dist_bound, stop, index, candidate, k + 1}
                else
                  d = Distance.squared_euclidean(train_data[j], query_data[i])
                  visited = Nx.indexed_put(visited, Nx.new_axis(j, 0), Nx.u8(1))

                  if d < dist_bound do
                    {query_indices, query_keys, query_flags} =
                      add_neighbor(
                        {query_indices, query_keys, query_flags},
                        i,
                        j,
                        d,
                        Nx.s8(-1)
                      )

                    search_candidates_indices =
                      Nx.indexed_put(
                        search_candidates_indices,
                        Nx.new_axis(search_ptr, 0),
                        j
                      )

                    search_candidates_distances =
                      Nx.indexed_put(search_candidates_distances, Nx.new_axis(search_ptr, 0), d)

                    search_ptr = search_ptr + 1

                    dist_bound =
                      (Nx.tensor(1.0, type: to_float_type(query_data)) + opts[:eps]) *
                        query_keys[[i, 0]]

                    {{query_indices, query_keys, query_flags},
                     {visited, search_candidates_indices, search_candidates_distances,
                      search_ptr}, query_data, {search_indices, search_keys, search_flags},
                     initial_candidates, initial_distances, train_data, i, dist_bound, stop,
                     index, candidate, k + 1}
                  else
                    {{query_indices, query_keys, query_flags},
                     {visited, search_candidates_indices, search_candidates_distances,
                      search_ptr}, query_data, {search_indices, search_keys, search_flags},
                     initial_candidates, initial_distances, train_data, i, dist_bound, stop,
                     index, candidate, k + 1}
                  end
                end
              end
            end

          {{query_indices, query_keys, query_flags},
           {visited, search_candidates_indices, search_candidates_distances, search_ptr},
           query_data, search_graph, initial_candidates, initial_distances, train_data, i,
           dist_bound, stop, index + 1}
        end

        {{query_indices, query_keys, query_flags}, query_data, search_graph, initial_candidates,
         initial_distances, train_data, rng_key, i + 1}
      end

    ord = Nx.argsort(query_keys, axis: 1, stable: true)
    query_indices = Nx.take_along_axis(query_indices, ord, axis: 1)
    query_keys = Nx.take_along_axis(query_keys, ord, axis: 1)
    {query_indices, query_keys, search_graph}
  end
end
