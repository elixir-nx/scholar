defmodule Scholar.Neighbors.Graph do
  import Nx.Defn
  require Nx
  alias Scholar.Neighbors.RandomProjectionForest, as: Forest

  knn_schema = [
    k: [
      required: true,
      type: :pos_integer,
      doc: "The number of neighbors in the graph."
    ]
  ]

  # TODO: Rename this
  approximate_knn_schema =
    knn_schema ++
      [
        min_leaf_size: [
          type: :pos_integer,
          doc: "The minimum number of points in ..."
        ],
        num_trees: [
          type: :pos_integer,
          doc: "The number of trees in random projection forest."
        ],
        num_iters: [
          type: :non_neg_integer,
          default: 1,
          doc: "The number of times to perform neighborhood expansion."
        ],
        key: [
          type: {:custom, Scholar.Options, :key, []},
          doc: """
          Used for random number generation in parameter initialization.
          If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
          """
        ]
      ]

  @knn_schema NimbleOptions.new!(knn_schema)
  @approximate_knn_schema NimbleOptions.new!(approximate_knn_schema)

  deftransform exact_brute_force(x, opts) do
    opts = NimbleOptions.validate!(opts, @knn_schema)
    exact_brute_force_n(x, opts)
  end

  defn exact_brute_force_n(x, opts) do
    k = opts[:k]
    # TODO: Add other metrics other than Euclidean
    x
    |> Nx.new_axis(1)
    |> Nx.subtract(Nx.new_axis(x, 0))
    |> Nx.pow(2)
    |> Nx.sum(axes: [2])
    |> Nx.put_diagonal(Nx.broadcast(:infinity, {Nx.axis_size(x, 0)}))
    |> Nx.negate()
    |> Nx.top_k(k: k)
    |> elem(1)
  end

  @doc """
  Constructs the approximate k-nearest neighbor graph using random projection forest
  and neighborhood expansion.
  """
  deftransform approximate_random_projection_forest_and_expansion(x, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}\
            """
    end

    opts = NimbleOptions.validate!(opts, @approximate_knn_schema)
    k = opts[:k]
    min_leaf_size = opts[:min_leaf_size]

    min_leaf_size =
      if min_leaf_size do
        if min_leaf_size > k, do: min_leaf_size, else: k + 1
      else
        max(10, 2 * k + 1)
      end

    size = Nx.axis_size(x, 0)
    num_trees = if opts[:num_trees], do: opts[:num_trees], else: 5 + round(:math.pow(size, 0.25))
    num_iters = opts[:num_iters]
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    forest = Forest.fit(x, num_trees: num_trees, min_leaf_size: min_leaf_size, key: key)
    IO.inspect(forest.leaf_size)
    IO.inspect(num_trees)
    leaves = Forest.predict(forest, x) |> Nx.reshape({size, num_trees * forest.leaf_size})
    graph = make_graph(leaves, x, k: k)
    expand(graph, x, num_iters: num_iters)
  end

  @doc """
  Performs neighborhood expansion.
  """
  defn expand(graph, x, opts) do
    num_iters = opts[:num_iters]

    {graph, _} =
      while {
              graph,
              {x, iter = 0}
            },
            iter < num_iters do
        {expansion_iter(graph, x), {x, iter + 1}}
      end

    graph
  end

  defnp expansion_iter(graph, x) do
    {size, k} = Nx.shape(graph)
    candidate_lists = Nx.take(graph, graph) |> Nx.reshape({size, k * k})
    candidate_lists = Nx.concatenate([graph, candidate_lists], axis: 1)

    make_graph(candidate_lists, x, k: k)
  end

  defnp make_graph(candidate_lists, x, opts) do
    k = opts[:k]
    {n, length} = Nx.shape(candidate_lists)

    index_mask =
      Nx.iota({n, 1})
      |> Nx.broadcast({n, length})
      |> Nx.equal(candidate_lists)

    sorted_indices = Nx.argsort(candidate_lists, axis: 1, stable: true)
    inverse = inverse_permutation(sorted_indices)
    sorted = Nx.take_along_axis(candidate_lists, sorted_indices, axis: 1)

    duplicate_mask =
      Nx.concatenate(
        [
          Nx.broadcast(0, {n, 1}),
          Nx.equal(sorted[[.., 0..-2//1]], sorted[[.., 1..-1//1]])
        ],
        axis: 1
      )
      |> Nx.take_along_axis(inverse, axis: 1)

    mask = index_mask or duplicate_mask

    distances =
      x
      |> Nx.new_axis(1)
      |> Nx.subtract(Nx.take(x, candidate_lists))
      |> Nx.pow(2)
      |> Nx.sum(axes: [-1])

    distances = Nx.select(mask, :infinity, distances)
    indices = Nx.argsort(distances, axis: -1)

    candidate_lists =
      Nx.take(
        Nx.vectorize(candidate_lists, :samples),
        Nx.vectorize(indices, :samples)
      )
      |> Nx.devectorize()
      |> Nx.rename(nil)

    Nx.slice_along_axis(candidate_lists, 0, k, axis: 1)
  end

  defnp inverse_permutation(indices) do
    {n, length} = Nx.shape(indices)
    target = Nx.broadcast(Nx.u32(0), {n, length})
    samples = Nx.iota({n, length, 1}, axis: 0)

    indices =
      Nx.concatenate([samples, Nx.new_axis(indices, 2)], axis: 2)
      |> Nx.reshape({n * length, 2})

    updates = Nx.iota({n, length}, axis: 1) |> Nx.reshape({n * length})
    Nx.indexed_add(target, indices, updates)
  end
end
