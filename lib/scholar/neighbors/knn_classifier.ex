defmodule Scholar.Neighbors.KNNClassifier do
  @moduledoc """
  K-Nearest Neighbors Classifier.

  Performs classification by computing the (weighted) majority voting among k-nearest neighbors.
  """

  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, keep: [:num_classes, :weights], containers: [:algorithm, :labels]}
  defstruct [:algorithm, :num_classes, :weights, :labels]

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
    num_classes: [
      required: true,
      type: :pos_integer,
      doc: "The number of possible classes."
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
  Fits a k-NN classifier model.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  Algorithm-specific options (e.g. `:num_neighbors`, `:metric`) should be provided together with the classifier options.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1])
      iex> model = Scholar.Neighbors.KNNClassifier.fit(x, y, num_neighbors: 3, num_classes: 2)
      iex> model.algorithm
      Scholar.Neighbors.BruteKNN.fit(x, num_neighbors: 3)
      iex> model.labels
      Nx.tensor([0, 0, 0, 1, 1])

      iex> x = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1])
      iex> model = Scholar.Neighbors.KNNClassifier.fit(x, y, algorithm: :kd_tree, num_neighbors: 3, metric: {:minkowski, 1}, num_classes: 2)
      iex> model.algorithm
      Scholar.Neighbors.KDTree.fit(x, num_neighbors: 3, metric: {:minkowski, 1})
      iex> model.labels
      Nx.tensor([0, 0, 0, 1, 1])

      iex> x = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1])
      iex> key = Nx.Random.key(12)
      iex> model = Scholar.Neighbors.KNNClassifier.fit(x, y, algorithm: :random_projection_forest, num_neighbors: 2, num_classes: 2, num_trees: 4, key: key)
      iex> model.algorithm
      Scholar.Neighbors.RandomProjectionForest.fit(x, num_neighbors: 2, num_trees: 4, key: key)
      iex> model.labels
      Nx.tensor([0, 0, 0, 1, 1])
  """
  deftransform fit(x, y, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected x to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}
            """
    end

    if Nx.rank(y) != 1 do
      raise ArgumentError,
            """
            expected y to have shape {num_samples}, \
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

    {opts, algorithm_opts} = Keyword.split(opts, [:algorithm, :num_classes, :weights])
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
      num_classes: opts[:num_classes],
      labels: y,
      weights: opts[:weights]
    }
  end

  @doc """
  Predicts classes using a k-NN classifier model.

  ## Examples

      iex> x_train = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y_train = Nx.tensor([0, 0, 0, 1, 1])
      iex> model = Scholar.Neighbors.KNNClassifier.fit(x_train, y_train, num_neighbors: 3, num_classes: 2)
      iex> x = Nx.tensor([[1, 3], [4, 2], [3, 6]])
      iex> Scholar.Neighbors.KNNClassifier.predict(model, x)
      Nx.tensor([0, 0, 1])
  """
  defn predict(model, x) do
    {neighbors, distances} = compute_knn(model.algorithm, x)
    neighbor_labels = Nx.take(model.labels, neighbors)

    case model.weights do
      :uniform ->
        Nx.mode(neighbor_labels, axis: 1)

      :distance ->
        weighted_mode(neighbor_labels, Scholar.Neighbors.Utils.check_weights(distances))
    end
  end

  @doc """
  Predicts class probabilities using a k-NN classifier model.

  ## Examples

      iex> x_train = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y_train = Nx.tensor([0, 0, 0, 1, 1])
      iex> model = Scholar.Neighbors.KNNClassifier.fit(x_train, y_train, num_neighbors: 3, num_classes: 2)
      iex> x = Nx.tensor([[1, 3], [4, 2], [3, 6]])
      iex> Scholar.Neighbors.KNNClassifier.predict_probability(model, x)
      Nx.tensor(
        [
          [1.0, 0.0],
          [1.0, 0.0],
          [0.3333333432674408, 0.6666666865348816]
        ]
      )
  """
  defn predict_probability(model, x) do
    num_samples = Nx.axis_size(x, 0)
    type = to_float_type(x)
    {neighbors, distances} = compute_knn(model.algorithm, x)
    neighbor_labels = Nx.take(model.labels, neighbors)
    proba = Nx.broadcast(Nx.tensor(0.0, type: type), {num_samples, model.num_classes})

    weights =
      case model.weights do
        :uniform -> Nx.broadcast(1.0, neighbors)
        :distance -> Scholar.Neighbors.Utils.check_weights(distances)
      end

    indices =
      Nx.stack(
        [Nx.iota(Nx.shape(neighbor_labels), axis: 0), neighbor_labels],
        axis: 2
      )
      |> Nx.flatten(axes: [0, 1])

    proba = Nx.indexed_add(proba, indices, Nx.flatten(weights))
    normalizer = Nx.sum(proba, axes: [1])
    normalizer = Nx.select(normalizer == 0, 1, normalizer)
    proba / Nx.new_axis(normalizer, 1)
  end

  deftransformp compute_knn(algorithm, x) do
    algorithm.__struct__.predict(algorithm, x)
  end

  defnp weighted_mode(tensor, weights) do
    tensor_size = Nx.size(tensor)

    cond do
      tensor_size == 1 ->
        Nx.squeeze(tensor, axes: [1])

      true ->
        weighted_mode_general(tensor, weights)
    end
  end

  defnp weighted_mode_general(tensor, weights) do
    {num_samples, num_features} = tensor_shape = Nx.shape(tensor)

    indices = Nx.argsort(tensor, axis: 1)

    sorted = Nx.take_along_axis(tensor, indices, axis: 1)

    size_to_broadcast = {num_samples, 1}

    group_indices =
      Nx.concatenate(
        [
          Nx.broadcast(0, size_to_broadcast),
          Nx.not_equal(
            Nx.slice_along_axis(sorted, 0, Nx.axis_size(sorted, 1) - 1, axis: 1),
            Nx.slice_along_axis(sorted, 1, Nx.axis_size(sorted, 1) - 1, axis: 1)
          )
        ],
        axis: 1
      )
      |> Nx.cumulative_sum(axis: 1)

    num_elements = Nx.size(tensor_shape)

    counting_indices =
      [
        Nx.shape(group_indices)
        |> Nx.iota(axis: 0)
        |> Nx.reshape({num_elements, 1}),
        Nx.reshape(group_indices, {num_elements, 1})
      ]
      |> Nx.concatenate(axis: 1)

    to_add = Nx.flatten(weights)

    indices =
      (indices + num_features * Nx.iota(tensor_shape, axis: 0))
      |> Nx.flatten()

    weights = Nx.take(to_add, indices)

    largest_group_indices =
      Nx.broadcast(0, sorted)
      |> Nx.indexed_add(counting_indices, weights)
      |> Nx.argmax(axis: 1, keep_axis: true)

    indices =
      largest_group_indices
      |> Nx.broadcast(group_indices)
      |> Nx.equal(group_indices)
      |> Nx.argmax(axis: 1, keep_axis: true)

    res = Nx.take_along_axis(sorted, indices, axis: 1)
    Nx.squeeze(res, axes: [1])
  end
end
