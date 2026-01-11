defmodule Scholar.Neighbors.LargeVis do
  @moduledoc """
  LargeVis algorithm for approximate k-nearest neighbor (k-NN) graph construction.

  The algorithms works in the following way. First, the approximate k-NN graph is constructed
  using a random projection forest. Then, the graph is refined by looking at the neighbors of
  neighbors of every point for a fixed number of iterations. This step is called NN-expansion.

  ## References

    * [Visualizing Large-scale and High-dimensional Data](https://arxiv.org/abs/1602.00370).
  """

  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Neighbors.RandomProjectionForest
  alias Scholar.Neighbors.Utils

  opts = [
    num_neighbors: [
      required: true,
      type: :pos_integer,
      doc: "The number of neighbors in the graph."
    ],
    metric: [
      type: {:in, [:squared_euclidean, :euclidean]},
      default: :euclidean,
      doc: "The function that measures distance between two points."
    ],
    min_leaf_size: [
      type: :pos_integer,
      doc: """
      The minimum number of points in every leaf.
      Must be at least num_neighbors.
      If not provided, it is set based on the number of neighbors.
      """
    ],
    num_trees: [
      type: :pos_integer,
      doc: """
      The number of trees in random projection forest.
      If not provided, it is set based on the dataset size.
      """
    ],
    num_iters: [
      type: :non_neg_integer,
      default: 3,
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

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Constructs the approximate k-NN graph with LargeVis.

  Returns neighbor indices and distances.

  ## Examples

      iex> key = Nx.Random.key(12)
      iex> tensor = Nx.iota({5, 2})
      iex> {graph, distances} = Scholar.Neighbors.LargeVis.fit(tensor, num_neighbors: 2, metric: :squared_euclidean, min_leaf_size: 2, num_trees: 3, key: key)
      iex> graph
      #Nx.Tensor<
        u32[5][2]
        [
          [0, 1],
          [1, 0],
          [2, 1],
          [3, 2],
          [4, 3]
        ]
      >
      iex> distances
      #Nx.Tensor<
        f32[5][2]
        [
          [0.0, 8.0],
          [0.0, 8.0],
          [0.0, 8.0],
          [0.0, 8.0],
          [0.0, 8.0]
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
    k = opts[:num_neighbors]

    metric =
      case opts[:metric] do
        :euclidean -> &Scholar.Metrics.Distance.euclidean/2
        :squared_euclidean -> &Scholar.Metrics.Distance.squared_euclidean/2
      end

    min_leaf_size = opts[:min_leaf_size] || max(10, 2 * k)

    size = Nx.axis_size(tensor, 0)
    num_trees = opts[:num_trees] || 5 + round(:math.pow(size, 0.25))
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)

    fit_n(
      tensor,
      key,
      num_neighbors: k,
      metric: metric,
      min_leaf_size: min_leaf_size,
      num_trees: num_trees,
      num_iters: opts[:num_iters]
    )
  end

  defnp fit_n(tensor, key, opts) do
    forest =
      RandomProjectionForest.fit(tensor,
        num_neighbors: opts[:num_neighbors],
        min_leaf_size: opts[:min_leaf_size],
        num_trees: opts[:num_trees],
        key: key
      )

    {graph, _} = RandomProjectionForest.predict(forest, tensor)
    expand(graph, tensor, metric: opts[:metric], num_iters: opts[:num_iters])
  end

  defn expand(graph, tensor, opts) do
    num_iters = opts[:num_iters]
    {n, k} = Nx.shape(graph)

    {result, _} =
      while {
              {
                graph,
                _distances = Nx.broadcast(Nx.tensor(:nan, type: to_float_type(tensor)), {n, k})
              },
              {tensor, iter = 0}
            },
            iter < num_iters do
        {expansion_iter(graph, tensor, metric: opts[:metric]), {tensor, iter + 1}}
      end

    result
  end

  defnp expansion_iter(graph, tensor, opts) do
    {size, k} = Nx.shape(graph)
    candidate_indices = Nx.take(graph, graph) |> Nx.reshape({size, k * k})
    candidate_indices = Nx.concatenate([graph, candidate_indices], axis: 1)

    Utils.brute_force_search_with_candidates(tensor, tensor, candidate_indices,
      num_neighbors: k,
      metric: opts[:metric]
    )
  end
end
