Nx.default_backend(EXLA.Backend)
key = Nx.Random.key(12)
alias Scholar.Neighbors.RandomProjectionForest, as: Forest
# tensor = Nx.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) |> Nx.new_axis(1)
# {x, key} = Nx.Random.normal(key, shape: 1000, 50})
# {x, key} = Nx.Random.uniform(key, shape: {68, 1})
{x, _key} = Nx.Random.uniform(key, shape: {135, 1})
# {n, d} = Nx.shape(x)
# k = 10
# forest1 = Forest.fit_bounded_n(x, min_leaf_size: 2, num_trees: 3, key: key)
# forest2 = Forest.fit(x, min_leaf_size: 2, num_trees: 3, key: key)
# IO.inspect(Nx.to_list(forest1.indices) == Nx.to_list(forest2.indices))

# bounded = Forest.fit_bounded(x, min_leaf_size: 1, num_trees: 1, key: key)
forest1 = Forest.fit_unbounded(x, min_leaf_size: 1, num_trees: 1, key: key)
forest2 = Forest.fit_bounded(x, min_leaf_size: 1, num_trees: 1, key: key)
IO.inspect(forest1.medians)
IO.inspect(forest2.medians)

# IO.inspect(forest)
# predictions = Forest.predict(forest, x)
# IO.inspect(predictions)

# sizes = 2..1000
# Enum.take_while(
#   sizes,
#   fn size ->
#     {x, _key} = Nx.Random.uniform(key, shape: {size, 1})
#     bounded = Forest.fit_bounded(x, min_leaf_size: 1, num_trees: 1, key: key)
#     unbounded = Forest.fit(x, min_leaf_size: 1, num_trees: 1, key: key)
#     Nx.to_list(bounded.indices) == Nx.to_list(unbounded.indices) and Nx.to_list(bounded.medians) == Nx.to_list(unbounded.medians)
#   end
# ) |> Enum.take(-1) |> IO.inspect(char_lists: false)

# graph = Scholar.Neighbors.RandomProjectionForest.knn_graph_construction(x, k: 10, num_trees: 15, min_leaf_size: 20, num_iters: 1, key: key)
# IO.inspect(graph)

# graph_brute_force =
#   Nx.new_axis(x, 1)
#   |> Nx.subtract(Nx.new_axis(x, 0))
#   |> Nx.pow(2)
#   |> Nx.sum(axes: [-1])
#   |> Nx.put_diagonal(Nx.broadcast(:infinity, {Nx.axis_size(x, 0)}))
#   |> Nx.negate()
#   |> Nx.top_k(k: k)
#   |> elem(1)

# accuracy =
#   graph
#   |> then(& Nx.concatenate([&1, graph_brute_force], axis: 1))
#   |> Nx.sort(axis: 1)
#   |> then(&(Nx.equal(&1[[.., 0..(2 * k - 2)]], &1[[.., 1..(2 * k - 1)]])))
#   |> Nx.sum()
#   |> Nx.divide(n * k)

# IO.inspect(accuracy)
