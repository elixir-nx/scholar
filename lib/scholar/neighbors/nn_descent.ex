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
  alias Scholar.Metrics.Distance

  @derive {Nx.Container, containers: [:neighbor_graph]}
  defstruct [:neighbor_graph]

  opts = [
    num_neighbors: [
      type: :pos_integer,
      default: 3,
      doc: "The number of neighbors to use in queries"
    ],
    max_candidates: [
      type: :pos_integer,
      default: 3,
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
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Builds a KDTree.

  ## Examples

      iex> Scholar.Neighbors.KDTree.fit(Nx.iota({5, 2}))
      %Scholar.Neighbors.KDTree{
        data: Nx.iota({5, 2}),
        levels: 3,
        indices: Nx.u32([3, 1, 4, 0, 2])
      }
  """
  deftransform fit(tensor, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    graph = initialize_graph(tensor, opts)
    nn_descent(tensor, graph, key, opts)
  end

  defnp initialize_graph(tensor, opts) do
    num_neighbors = opts[:num_neighbors]
    num_samples = Nx.axis_size(tensor, 0)

    {Nx.broadcast(-1, {num_samples, num_neighbors}),
     Nx.broadcast(Nx.Constants.max_finite(:f32), {num_samples, num_neighbors}),
     Nx.broadcast(0, {num_samples, num_neighbors})}
  end

  defnp get_size(indices, index0) do
    if(Nx.any(indices[index0] == -1)) do
      Nx.argmax(indices[index0] == -1)
    else
      Nx.axis_size(indices, 1)
    end
  end

  defnp init_random(data, curr_graph, num_neighbros, dist, opts) do
    rng_key = opts[:rng]
    new = 1

    {indices, keys, flags} = curr_graph
    num_heaps = Nx.axis_size(indices, 0)

    while {{indices, keys, flags}, {index0 = 0, data, rng_key, dist, new}}, index0 < num_heaps do
      missing = num_neighbros - get_size(indices, index0)

      {{indices, keys, flags}, _} =
        while {{indices, keys, flags}, {index0, j = 0, data, rng_key, dist}}, j < missing do
          {index1, new_rng_key} = Nx.Random.randint(rng_key, 1, num_heaps + 1, type: :u32)
          # check
          # index1 = Nx.reminder(randint, num_heaps)

          ### TO IMPLEMENT: dist
          # d = dist(data[index0], data[index1])
          d = Distance.squared_euclidean(data[index0], data[index1])
          {indices, keys, flags} = add_neighbor({indices, keys, flags}, index0, index1, d, new)

          {{indices, keys, flags}, {index0, j + 1, data, new_rng_key, dist, new}}
        end

      {{indices, keys, flags}, {index0 + 1, data, rng_key, dist, new}}
    end

    {indices, keys, flags}
  end

  defnp add_zero_nodes(curr_graph) do
    new = 1

    {indices, _keys, _flags} = curr_graph
    num_heaps = Nx.axis_size(indices, 0)

    while {i = 0, curr_graph, new}, i < num_heaps do
      curr_graph = add_neighbor(curr_graph, i, i, 0.0, new)
      {i + 1, curr_graph, new}
    end

    curr_graph
  end

  ### TO IMPLEMENT: Vectorization
  defnp update_by_leaves(data, curr_graph, leaves) do
    num_leaves = Nx.axis_size(leaves, 0)
    leaf_size = Nx.axis_size(leaves, 1)

    updates_index = 0
    updates_indices = Nx.broadcast(Nx.s64(0), {num_leaves, 2})
    updates_dist = Nx.broadcast(Nx.f32(0.0), {num_leaves})

    {_indices, keys, _flags} = curr_graph

    while {{updates_indices, updates_dist, updates_index}, {i = 0, keys}}, i < num_leaves do
      {{updates_indices, updates_dist, updates_index}, _} =
        while {{updates_indices, updates_dist, updates_index}, {i, j = 0, keys, stop = 0}},
              j < leaf_size and not stop do
          index0 = leaves[[i, j]]

          if index0 != -1 do
            {{updates_indices, updates_dist, updates_index}, _} =
              while {{updates_indices, updates_dist, updates_index},
                     {i, j, k = j + 1, keys, index0, stop_inner = 0}},
                    k < leaf_size and not stop_inner do
                index1 = leaves[[i, k]]

                if index1 != -1 do
                  ### TO IMPLEMENT: dist
                  # d = dist(data[index0], data[index1])
                  d = Distance.squared_euclidean(data[index0], data[index1])

                  {updates_indices, updates_dist, updates_index} =
                    if d < keys[[index0, 0]] or d < keys[[index1, 0]] do
                      updates_indices =
                        Nx.put_slice(
                          updates_indices,
                          updates_index,
                          Nx.concatenate([index0, index1])
                        )

                      updates_dist = Nx.indexed_put(updates_dist, updates_index, d)
                      {updates_indices, updates_dist, updates_index + 1}
                    else
                      {updates_indices, updates_dist, updates_index}
                    end

                  {{updates_indices, updates_dist, updates_index},
                   {i, j, k + 1, keys, index0, Nx.u8(0)}}
                else
                  {{updates_indices, updates_dist, updates_index},
                   {i, j, k + 1, keys, index0, Nx.u8(1)}}
                end
              end

            {{updates_indices, updates_dist, updates_index}, {i, j + 1, keys, Nx.u8(0)}}
          else
            {{updates_indices, updates_dist, updates_index}, {i, j + 1, keys, Nx.u8(1)}}
          end
        end

      {{updates_indices, updates_dist, updates_index}, {i + 1, keys}}
    end

    {curr_graph, _} =
      while {curr_graph, {i = 0, updates_index, updates_indices, updates_dist, new = 1}},
            i < updates_index do
        index0 = updates_indices[[i, 0]]
        index1 = updates_indices[[i, 1]]
        d = updates_dist[i]

        ### TO IMPLEMENT: add_neighbor
        curr_graph = add_neighbor(curr_graph, index0, index1, d, new)
        {curr_graph, {i + 1, updates_index, updates_indices, updates_dist, new}}
      end

    curr_graph
  end

  defnp sample_candidate(curr_graph, new_candidates, old_candidates, rng_key) do
    # new = 1
    # old = 0

    {indices, keys, flags} = curr_graph
    {num_heaps, num_nodes} = Nx.shape(indices)

    {{{indices, keys, flags}, new_candidates, old_candidates, rng_key}, _} =
      while {{{indices, keys, flags}, new_candidates, old_candidates, rng_key}, i = 0},
            i < num_heaps do
        {{{indices, keys, flags}, new_candidates, old_candidates, rng_key}, _} =
          while {{{indices, keys, flags}, new_candidates, old_candidates, rng_key}, {j = 0, i}},
                j < num_nodes do
            index1 = indices[[i, j]]
            flag = flags[[i, j]]

            if index1 == -1 do
              {{{indices, keys, flags}, new_candidates, old_candidates, rng_key}, {j + 1, i}}
            else
              # TODO: check scope of rng
              {priority, new_rng_key} = Nx.Random.randint(rng_key, 1, 100_000, type: :u32)

              if flag == Nx.u8(1) do
                new_candidates = add_neighbor(new_candidates, i, index1, priority, -1)
                new_candidates = add_neighbor(new_candidates, index1, i, priority, -1)

                {{{indices, keys, flags}, new_candidates, old_candidates, new_rng_key},
                 {j + 1, i}}
              else
                old_candidates = add_neighbor(old_candidates, i, index1, priority, -1)
                old_candidates = add_neighbor(old_candidates, index1, i, priority, -1)

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
      while {{{indices, keys, flags}, new_candidates_indices}, i = 0}, i < num_heaps do
        {{{indices, keys, flags}, new_candidates_indices}, _} =
          while {{{indices, keys, flags}, new_candidates_indices}, {j = 0, i}}, j < num_nodes do
            index1 = indices[[i, j]]
            termination = Nx.u8(0)

            {{{indices, keys, flags}, new_candidates_indices}, _} =
              while {{{indices, keys, flags}, new_candidates_indices},
                     {termination, index1, k = 0, i, j}},
                    k < new_candidates_num_nodes and not termination do
                {{indices, keys, flags}, termination} =
                  if new_candidates_indices[[i, k]] == index1 do
                    flags =
                      Nx.indexed_put(
                        flags,
                        Nx.concatenate([Nx.new_axis(i, 0), Nx.new_axis(j, 0)]),
                        Nx.u8(0)
                      )

                    {{indices, keys, flags}, Nx.u8(1)}
                  else
                    {{indices, keys, flags}, Nx.u8(0)}
                  end

                {{{indices, keys, flags}, new_candidates_indices},
                 {termination, index1, k + 1, i, j}}
              end

            {{{indices, keys, flags}, new_candidates_indices}, {j + 1, i}}
          end

        {{{indices, keys, flags}, new_candidates_indices}, i + 1}
      end

    {{indices, keys, flags}, new_candidates, old_candidates, rng_key}
  end

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
    update_index = 0
    updates_indices = Nx.broadcast(Nx.s64(0), {num_samples, 2})
    updates_dist = Nx.broadcast(Nx.f32(0.0), {num_samples})

    # {{_keys, _new_candidates_indices, _old_candidates_indices, updates_indices,
    #   updates_dist, update_index}, _} =
    while {{keys, new_candidates_indices, old_candidates_indices, updates_indices, updates_dist,
            update_index}, {i = 0, data}},
          i < size_new do
      {{keys, new_candidates_indices, old_candidates_indices, updates_indices, updates_dist,
        update_index}, {i + 1, data}}
    end

    # {{_, _, _, updates_indices, updates_dist, update_index}, _} =
    #   while {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #           updates_dist, update_index}, {j = 0, data, i}},
    #         j < new_candidates_num_nodes do
    #     index0 = new_candidates_indices[[i, j]]

    #     if index0 == -1 do
    #       {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #         updates_dist, update_index}, {j + 1, data, i}}
    #     else
    #       {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #         updates_dist, update_index},
    #        _} =
    #         while {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #                 updates_dist, update_index}, {k = j + 1, data, i, j, index0}},
    #               k < new_candidates_num_nodes do
    #           index1 = new_candidates_indices[[i, k]]

    #           if index1 == -1 do
    #             {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #               updates_dist, update_index}, {k + 1, data, i, j, index0}}
    #           else
    #             ### TO IMPLEMENT: dist
    #             d = Distance.squared_euclidean(data[index0], data[index1])
    #             # d = dist(data[index0], data[index1])

    #             {updates_indices, updates_dist, update_index} =
    #               if d < keys[[index0, 0]] or d < keys[[index1, 0]] do
    #                 updates_indices =
    #                   Nx.put_slice(
    #                     updates_indices,
    #                     update_index,
    #                     Nx.concatenate([Nx.new_axis(index0, 0), Nx.new_axis(index1, 0)])
    #                   )

    #                 updates_dist = Nx.indexed_put(updates_dist, update_index, d)
    #                 {updates_indices, updates_dist, update_index + 1}
    #               else
    #                 {updates_indices, updates_dist, update_index}
    #               end

    #             {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #               updates_dist, update_index}, {k + 1, data, i, j, index0}}
    #           end
    #         end

    #       {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #         updates_dist, update_index},
    #        _} =
    #         while {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #                 updates_dist, update_index}, {k = 0, data, i, index0}},
    #               k < old_candidates_num_nodes do
    #           index1 = old_candidates_indices[[i, k]]

    #           if index1 == -1 do
    #             {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #               updates_dist, update_index}, {k + 1, data, i, index0}}
    #           else
    #             ### TO IMPLEMENT: dist
    #             # d = dist(data[index0], data[index1])
    #             d = Distance.squared_euclidean(data[index0], data[index1])

    #             {updates_indices, updates_dist, update_index} =
    #               if d < keys[[index0, 0]] or d < keys[[index1, 0]] do
    #                 updates_indices =
    #                   Nx.put_slice(
    #                     updates_indices,
    #                     update_index,
    #                     Nx.concatenate([Nx.new_axis(index0, 0), Nx.new_axis(index1, 0)])
    #                   )

    #                 updates_dist = Nx.indexed_put(updates_dist, update_index, d)
    #                 {updates_indices, updates_dist, update_index + 1}
    #               else
    #                 {updates_indices, updates_dist, update_index}
    #               end

    #             {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #               updates_dist, update_index}, {k + 1, data, i, index0}}
    #           end
    #         end

    #       {{keys, new_candidates_indices, old_candidates_indices, updates_indices,
    #         updates_dist, update_index}, {j + 1, data, i}}
    #     end
    #   end

    #   {{keys, new_candidates_indices, old_candidates_indices, updates_indices, updates_dist,
    #     update_index}, {i + 1, data}}
    # end

    {updates_indices, updates_dist, update_index}
  end

  defnp apply_updates(current_graph, updates_indices, updates_dist, update_index) do
    new = 1

    {current_graph, _} =
      while {current_graph, {i = 0, update_index, updates_indices, updates_dist, new}},
            i < update_index do
        index0 = updates_indices[[i, 0]]
        index1 = updates_indices[[i, 1]]
        d = updates_dist[i]

        ### TO IMPLEMENT: add_neighbor
        current_graph = add_neighbor(current_graph, index0, index1, d, new)
        current_graph = add_neighbor(current_graph, index1, index0, d, new)
        {current_graph, {i + 1, update_index, updates_indices, updates_dist, new}}
      end

    current_graph
  end

  defnp nn_descent(data, curr_graph, key, opts \\ []) do
    max_iters = opts[:max_iterations]
    tol = opts[:tol]
    max_candidates = opts[:max_candidates]
    num_neighbors = opts[:num_neighbors]
    num_samples = Nx.axis_size(data, 0)
    termination = Nx.u8(0)

    {curr_graph, _} =
      while {curr_graph, {i = 0, termination, key}}, i < max_iters and not termination do
        new_candidates =
          {Nx.broadcast(-1, {num_samples, max_candidates}),
           Nx.broadcast(Nx.Constants.max_finite(:s64), {num_samples, max_candidates}),
           Nx.broadcast(0, {num_samples, max_candidates})}

        old_candidates =
          {Nx.broadcast(-1, {num_samples, max_candidates}),
           Nx.broadcast(Nx.Constants.max_finite(:s64), {num_samples, max_candidates}),
           Nx.broadcast(0, {num_samples, max_candidates})}

        {curr_graph, new_candidates, old_candidates, key} =
          sample_candidate(curr_graph, new_candidates, old_candidates, key)

        {updates_indices, updates_dist, update_index} =
          generate_graph_updates(data, curr_graph, new_candidates, old_candidates)

        # curr_graph = apply_updates(curr_graph, updates_indices, updates_dist, update_index)

        # termination =
        #   if update_index < tol * num_samples * num_neighbors do
        #     Nx.u8(1)
        #   else
        #     Nx.u8(0)
        #   end
        {curr_graph, {i + 1, termination, key}}
      end

    curr_graph
  end

  defn get_leaves_from_forest(forest) do
    leaf_size = forest.leaf_size
    {num_trees, num_indices} = Nx.shape(forest.indices)
    to_concat = rem(num_indices, leaf_size)

    leaves =
      if to_concat != 0 do
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
        while {{indices, keys, flags, curr = 0, swap = 0}, {index0, stop, key, flag}},
              not stop do
          left_child = 2 * curr + 1
          right_child = left_child + 1

          # to check if this simplification is correct
          {swap, stop} =
            cond do
              left_child < num_nodes ->
                {swap, stop}

              right_child >= num_nodes and keys[[index0, left_child]] > key ->
                {left_child, stop}

              keys[[index0, left_child]] >= keys[[index0, right_child]] and
                  keys[[index0, left_child]] > key ->
                {left_child, stop}

              keys[[index0, right_child]] > key ->
                {right_child, stop}

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
            if flag != -1,
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
        if flag != -1,
          do: Nx.indexed_put(flags, Nx.concatenate([index0_ext, curr_ext]), flag),
          else: flags

      {indices, keys, flags}
    end
  end
end
