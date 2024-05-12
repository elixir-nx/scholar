defmodule Scholar.Neighbors.KNNClassifier do
  @moduledoc """
  K-Nearest Neighbors Classifier.

  Performs classifiction by looking at the k-nearest neighbors of a point and using (weighted) majority voting.
  """

  import Nx.Defn
  require Nx

  @derive {Nx.Container, keep: [:algorithm, :num_classes, :weights], containers: [:labels]}
  defstruct [:algorithm, :num_classes, :weights, :labels]

  opts = [
    algorithm: [
      type: {:or, [:atom, {:tuple, [:atom, :keyword_list]}]},
      default: :brute,
      doc: """
      k-NN algorithm to be used for finding the nearest neighbors. It can be provided as
      an atom or a tuple containing an atom and algorithm specific options.
      Possible values for the atom:

        * `:brute` - Brute-force search. See `Scholar.Neighbors.BruteKNN` for more details.

        * `:kd_tree` - k-d tree. See `Scholar.Neighbors.KDTree` for more details.

        * `:random_projection_forest` - Random projection forest. See `Scholar.Neighbors.RandomProjectionForest` for more details.

        * Module implementing fit/2 and predict/2.
      """
    ],
    num_neighbors: [
      required: true,
      type: :pos_integer,
      doc: "The number of nearest neighbors."
    ],
    metric: [
      type: {:or, [{:custom, Scholar.Options, :metric, []}, {:fun, 2}]},
      default: {:minkowski, 2},
      doc: """
      The function that measures distance between two points. Possible values:

        * `{:minkowski, p}` - Minkowski metric. By changing value of `p` parameter (a positive number or `:infinity`)
        we can set Manhattan (`1`), Euclidean (`2`), Chebyshev (`:infinity`), or any arbitrary $L_p$ metric.

        * `:cosine` - Cosine metric.

      Keep in mind that different algorithms support different metrics. For more information have a look at the corresponding modules.
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

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y = Nx.tensor([0, 0, 0, 1, 1])
      iex> model = Scholar.Neighbors.KNNClassifier.fit(x, y, num_neighbors: 3, num_classes: 2)
      iex> model.algorithm
      Scholar.Neighbors.BruteKNN.fit(x, num_neighbors: 3)
      iex> model.labels
      Nx.tensor([0, 0, 0, 1, 1])
  """
  deftransform fit(x, y, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {num_samples, num_features},
             got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    if Nx.rank(y) != 1 and Nx.axis_size(x, 0) == Nx.axis_size(y, 0) do
      raise ArgumentError,
            "expected y to have shape {num_samples},
             got tensor with shape: #{inspect(Nx.shape(y))}"
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)

    {algorithm_name, algorithm_opts} =
      if is_atom(opts[:algorithm]) do
        {opts[:algorithm], []}
      else
        opts[:algorithm]
      end

    knn_module =
      case algorithm_name do
        :brute ->
          Scholar.Neighbors.BruteKNN

        :kd_tree ->
          Scholar.Neighbors.KDTree

        :random_projection_forest ->
          Scholar.Neighbors.RandomProjectionForest

        knn_module when is_atom(knn_module) ->
          knn_module

        _ ->
          raise ArgumentError,
                """
                not supported
                """
      end

    # TODO: Maybe raise an error if :num_neighbors or :metric is already in algorithm_opts?

    algorithm_opts = Keyword.put(algorithm_opts, :num_neighbors, opts[:num_neighbors])
    algorithm_opts = Keyword.put(algorithm_opts, :metric, opts[:metric])

    algorithm = knn_module.fit(x, algorithm_opts)

    %__MODULE__{
      algorithm: algorithm,
      num_classes: opts[:num_classes],
      labels: y,
      weights: opts[:weights]
    }
  end

  @doc """
  Makes predictions using a k-NN classifier model.

  ## Examples

      iex> x_train = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> y_train = Nx.tensor([0, 0, 0, 1, 1])
      iex> model = Scholar.Neighbors.KNNClassifier.fit(x_train, y_train, num_neighbors: 3, num_classes: 2)
      iex> x_test = Nx.tensor([[1, 3], [4, 2], [3, 6]])
      iex> Scholar.Neighbors.KNNClassifier.predict(model, x_test)
      Nx.tensor([0, 0, 1])
  """
  deftransform predict(model, x) do
    knn_module = model.algorithm.__struct__
    {neighbors, distances} = knn_module.predict(model.algorithm, x)
    labels_pred = Nx.take(model.labels, neighbors)

    case model.weights do
      :uniform -> Nx.mode(labels_pred, axis: 1)
      :distance -> weighted_mode(labels_pred, check_weights(distances))
    end
  end

  defnp check_weights(weights) do
    zero_mask = weights == 0
    zero_rows = zero_mask |> Nx.any(axes: [1], keep_axes: true) |> Nx.broadcast(weights)
    weights = Nx.select(zero_mask, 1, weights)
    weights_inv = 1 / weights
    Nx.select(zero_rows, Nx.select(zero_mask, 1, 0), weights_inv)
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
