defmodule Scholar.Neighbors.KNNRegressor do
  @moduledoc """
  K-Nearest Neighbors Regressor.

  Performs regression by computing the (weighted) mean of k-nearest neighbor labels.
  """
  import Nx.Defn

  @derive {Nx.Container, keep: [:weights], containers: [:algorithm, :labels]}
  defstruct [:algorithm, :weights, :labels]

  opts = [
    algorithm: [
      type: :atom,
      default: :brute,
      doc: """
      Algorithm used to compute the k-nearest neighbors. Possible values:

        * `:brute` - Brute-force search. See `Scholar.Neighbors.BruteKNN` for more details.

        * `:kd_tree` - k-d tree. See `Scholar.Neighbors.KDTree` for more details.

        * `:random_projection_forest` - Random projection forest. See `Scholar.Neighbors.RandomProjectionForest` for more details.

        * Module implementing `fit(data, opts)` and `predict(model, query)`. predict/2 must return a tuple containing indices
        of k-nearest neighbors of query points as well as distances between query points and their k-nearest neighbors.
      """
    ],
    weights: [
      type: {:in, [:uniform, :distance]},
      default: :uniform,
      doc: """
      Weight function used in prediction. Possible values:

        * `:uniform` - uniform weights. All points in each neighborhood are weighted equally.

        * `:distance` - weight points by the inverse of their distance. in this case, closer neighbors of
          a query point will have a greater influence than neighbors which are further away.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a k-NN regressor model.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  Algorithm-specific options (e.g. `:num_neighbors`, `:metric`) should be provided together with the regressor options.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y = Nx.tensor([[1], [2], [3], [4], [5]])
      iex> model = Scholar.Neighbors.KNNRegressor.fit(x, y, num_neighbors: 3)
      iex> model.algorithm
      Scholar.Neighbors.BruteKNN.fit(x, num_neighbors: 3)
      iex> model.labels
      Nx.tensor([[1], [2], [3], [4], [5]])

      iex> x = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y = Nx.tensor([[1], [2], [3], [4], [5]])
      iex> model = Scholar.Neighbors.KNNRegressor.fit(x, y, algorithm: :kd_tree, num_neighbors: 3, metric: {:minkowski, 1})
      iex> model.algorithm
      Scholar.Neighbors.KDTree.fit(x, num_neighbors: 3, metric: {:minkowski, 1})
      iex> model.labels
      Nx.tensor([[1], [2], [3], [4], [5]])

      iex> x = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y = Nx.tensor([[1], [2], [3], [4], [5]])
      iex> key = Nx.Random.key(12)
      iex> model = Scholar.Neighbors.KNNRegressor.fit(x, y, algorithm: :random_projection_forest, num_neighbors: 2, num_trees: 4, key: key)
      iex> model.algorithm
      Scholar.Neighbors.RandomProjectionForest.fit(x, num_neighbors: 2, num_trees: 4, key: key)
      iex> model.labels
      Nx.tensor([[1], [2], [3], [4], [5]])
  """
  deftransform fit(x, y, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected x to have shape {num_samples, num_features_in}, \
            got tensor with shape: #{inspect(Nx.shape(x))}
            """
    end

    if Nx.rank(y) != 2 do
      raise ArgumentError,
            """
            expected y to have shape {num_samples, num_features_out}, \
            got tensor with shape: #{inspect(Nx.shape(y))}
            """
    end

    if Nx.axis_size(x, 0) != Nx.axis_size(y, 0) do
      raise ArgumentError,
            """
            expected x and y to have the same first dimension, \
            got #{Nx.axis_size(x, 0)} and #{Nx.axis_size(y, 0)}
            """
    end

    {opts, algorithm_opts} = Keyword.split(opts, [:algorithm, :weights])
    opts = NimbleOptions.validate!(opts, @opts_schema)

    algorithm_module =
      case opts[:algorithm] do
        :brute ->
          Scholar.Neighbors.BruteKNN

        :kd_tree ->
          Scholar.Neighbors.KDTree

        :random_projection_forest ->
          Scholar.Neighbors.RandomProjectionForest

        module when is_atom(module) ->
          module
      end

    algorithm = algorithm_module.fit(x, algorithm_opts)

    %__MODULE__{
      algorithm: algorithm,
      labels: y,
      weights: opts[:weights]
    }
  end

  @doc """
  Predicts labels using a k-NN regressor model.

  ## Examples

      iex> x_train = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y_train = Nx.tensor([[1], [2], [3], [4], [5]])
      iex> model = Scholar.Neighbors.KNNRegressor.fit(x_train, y_train, num_neighbors: 3)
      iex> x = Nx.tensor([[1, 3], [4, 2], [3, 6]])
      iex> Scholar.Neighbors.KNNRegressor.predict(model, x)
      Nx.tensor([[2.0], [2.0], [4.0]])
  """
  defn predict(model, x) do
    {neighbors, distances} = compute_knn(model.algorithm, x)
    neighbor_labels = Nx.take(model.labels, neighbors)

    case model.weights do
      :uniform ->
        Nx.mean(neighbor_labels, axes: [1])

      :distance ->
        weights =
          Scholar.Neighbors.Utils.check_weights(distances)
          |> Nx.new_axis(2)
          |> Nx.broadcast(neighbor_labels)

        Nx.weighted_mean(neighbor_labels, weights, axes: [1])
    end
  end

  deftransformp compute_knn(algorithm, x) do
    algorithm.__struct__.predict(algorithm, x)
  end
end
