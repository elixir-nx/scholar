# mix run benchmarks/knn.exs
Nx.global_default_backend(EXLA.Backend)
Nx.Defn.global_default_options(compiler: EXLA)

key = Nx.Random.key(System.os_time())

inputs_knn = %{
  "100x10" => elem(Nx.Random.uniform(key, 0, 100, shape: {100, 10}), 0),
  "1000x10" => elem(Nx.Random.uniform(key, 0, 1000, shape: {1000, 10}), 0),
  "10000x10" => elem(Nx.Random.uniform(key, 0, 10000, shape: {10000, 10}), 0)
}

Benchee.run(
  %{
    "kdtree" => fn x ->
      kdtree = Scholar.Neighbors.KDTree.fit(x)
      Scholar.Neighbors.KDTree.predict(kdtree, x, k: 4)
    end,
    "brute force knn" => fn x ->
      model =
        Scholar.Neighbors.KNearestNeighbors.fit(x, Nx.broadcast(1, {Nx.axis_size(x, 0)}),
          num_classes: 2,
          num_neighbors: 4
        )

      Scholar.Neighbors.KNearestNeighbors.k_neighbors(model, x)
    end
  },
  time: 10,
  memory_time: 2,
  inputs: inputs_knn
)
