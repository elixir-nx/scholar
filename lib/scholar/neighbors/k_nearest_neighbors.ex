defmodule Scholar.Neighbors.KNearestNeighbors do
  @moduledoc """
  The K-Nearest Neighbors.

  It implements both classification and regression. This implements the linear
  version of kNN and therefore it has time complexity $O(N^2)$ for $N$ samples.
  """
  import Nx.Defn
  import Scholar.Shared
  require Nx

  @derive {Nx.Container,
           keep: [:default_num_neighbors, :weights, :num_classes, :task, :metric],
           containers: [:data, :labels]}
  defstruct [:data, :labels, :default_num_neighbors, :weights, :num_classes, :task, :metric]

  opts = [
    num_neighbors: [
      type: :pos_integer,
      default: 5,
      doc: "The number of neighbors to use by default for `k_neighbors` queries"
    ],
    num_classes: [
      type: :pos_integer,
      doc: "Number of classes in provided labels"
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
    ],
    metric: [
      type: {:custom, Scholar.Options, :metric, []},
      default: {:minkowski, 2},
      doc: ~S"""
      Name of the metric. Possible values:

      * `{:minkowski, p}` - Minkowski metric. By changing value of `p` parameter (a positive number or `:infinity`)
        we can set Manhattan (`1`), Euclidean (`2`), Chebyshev (`:infinity`), or any arbitrary $L_p$ metric.

      * `:cosine` - Cosine metric.
      """
    ],
    task: [
      type: {:in, [:classification, :regression]},
      default: :classification,
      doc: """
      Task that will be performed using K Nearest Neighbors. Possible values:

      * `:classification` - Classifier implementing the k-nearest neighbors vote.

      * `:regression` - Regression based on K-Nearest Neighbors.
        The target is predicted by local interpolation of the targets associated of the nearest neighbors in the training set.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fit the K-nearest neighbors classifier from the training data set.

  For classification, provided labels needs to be consecutive natural numbers. If your labels does
  not meet this condition please use `Scholar.Preprocessing.ordinal_encode`

  Currently 2D labels only supported for regression task.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:data` - Training data.

    * `:labels` - Labels of each point.

    * `:default_num_neighbors` - The number of neighbors to use by default for `k_neighbors` queries.

    * `:weights` - Weight function used in prediction.

    * `:num_classes` - Number of classes in provided labels.

    * `:task` - Task that will be performed using K-Nearest Neighbors.
      For `:classification` task, model will be a classifier implementing the K-Nearest Neighbors vote.
      For `:regression` task, model is a regressor based on K-Nearest Neighbors.

    * `:metric` - Name of the metric.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> Scholar.Neighbors.KNearestNeighbors.fit(x, y, num_classes: 2)
      %Scholar.Neighbors.KNearestNeighbors{
        data: Nx.tensor(
          [
            [1, 2],
            [2, 4],
            [1, 3],
            [2, 5]
          ]
        ),
        labels: Nx.tensor(
          [1, 0, 1, 1]
        ),
        default_num_neighbors: 4,
        weights: :uniform,
        num_classes: 2,
        task: :classification,
        metric: {:minkowski, 2}
      }
  """
  deftransform fit(x, y, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected input tensor to have shape {n_samples, n_features} or {num_samples, num_samples},
             got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    if Nx.rank(y) > 2 do
      raise ArgumentError,
            "expected labels to have shape {num_samples} or {num_samples, num_outputs},
            got tensor with shape: #{inspect(Nx.shape(y))}"
    end

    {num_samples, _} = Nx.shape(x)
    num_targets = Nx.axis_size(y, 0)

    if num_samples != num_targets do
      raise ArgumentError,
            "expected labels to have the same size of the first axis as data,
      got: #{inspect(num_samples)} != #{inspect(num_targets)}"
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)

    if opts[:num_classes] == nil and opts[:task] == :classification do
      raise ArgumentError,
            "expected :num_classes to be provided for task :classification"
    end

    %__MODULE__{
      data: x,
      labels: y,
      default_num_neighbors: min(opts[:num_neighbors], num_samples),
      weights: opts[:weights],
      num_classes: opts[:num_classes],
      task: opts[:task],
      metric: opts[:metric]
    }
  end

  @doc """
  Makes predictions with the given `model` on inputs `x`.

  ## Return Values

  It returns a tensor with predicted class labels.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> model = Scholar.Neighbors.KNearestNeighbors.fit(x, y, num_classes: 2)
      iex> Scholar.Neighbors.KNearestNeighbors.predict(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      Nx.tensor(
        [1, 1], type: :s64
      )
  """
  defn predict(%__MODULE__{labels: labels, weights: weights, task: task} = model, x) do
    {neigh_distances, neigh_indices} = k_neighbors(model, x)
    pred_labels = Nx.take(labels, neigh_indices)

    case task do
      :classification ->
        case weights do
          :distance -> mode_weighted(pred_labels, check_weights(neigh_distances), axis: 1)
          :uniform -> Nx.mode(pred_labels, axis: 1)
        end

      :regression ->
        case weights do
          :distance ->
            if Nx.rank(labels) == 2,
              do:
                Nx.weighted_mean(
                  pred_labels,
                  check_weights(neigh_distances)
                  |> Nx.new_axis(-1)
                  |> Nx.broadcast(pred_labels),
                  axes: [1]
                ),
              else: Nx.weighted_mean(pred_labels, check_weights(neigh_distances), axes: [1])

          :uniform ->
            Nx.mean(pred_labels, axes: [1])
        end
    end
  end

  @doc """
  Return probability estimates for the test data `x`.

  ## Return Values

  It returns a tensor with probabilities of classes. They are arranged in lexicographic order.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> model = Scholar.Neighbors.KNearestNeighbors.fit(x, y, num_classes: 2)
      iex> Scholar.Neighbors.KNearestNeighbors.predict_proba(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      Nx.tensor(
        [
          [0.75, 0.25],
          [0.75, 0.25]
        ]
      )
  """
  deftransform predict_proba(
                 %__MODULE__{
                   task: :classification
                 } = model,
                 x
               ) do
    predict_proba_n(model, x)
  end

  defnp predict_proba_n(
          %__MODULE__{
            labels: labels,
            weights: weights,
            num_classes: num_classes
          } = model,
          x
        ) do
    {num_samples, _} = Nx.shape(x)
    {neigh_distances, neigh_indices} = k_neighbors(model, x)
    pred_labels = Nx.take(labels, neigh_indices)
    x_type = Nx.Type.merge(to_float_type(x), {:f, 32})
    proba = Nx.broadcast(Nx.tensor(0.0, type: x_type), {num_samples, num_classes})

    weights_vals =
      case weights do
        :distance -> check_weights(neigh_distances)
        :uniform -> Nx.broadcast(1.0, neigh_indices)
      end

    indices =
      Nx.stack(
        [Nx.iota(Nx.shape(pred_labels), axis: 0), Nx.take(labels, pred_labels)],
        axis: -1
      )
      |> Nx.flatten(axes: [0, 1])

    proba = Nx.indexed_add(proba, indices, Nx.flatten(weights_vals))
    normalizer = Nx.sum(proba, axes: [1])
    normalizer = Nx.select(normalizer == 0, 1, normalizer)
    proba / Nx.new_axis(normalizer, -1)
  end

  @doc """
  Find the K-neighbors of a point.

  ## Return Values

    Returns indices of and distances to the neighbors of each point.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> model = Scholar.Neighbors.KNearestNeighbors.fit(x, y, num_classes: 2)
      iex> Scholar.Neighbors.KNearestNeighbors.k_neighbors(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      {Nx.tensor(
        [
          [0.3162279427051544, 0.7071065902709961, 1.5811389684677124, 2.469817876815796],
          [0.10000002384185791, 1.0049875974655151, 2.193171262741089, 3.132091760635376]
        ]
      ),
      Nx.tensor(
        [
          [1, 3, 2, 0],
          [0, 2, 1, 3]
        ]
      )}
  """
  defn k_neighbors(
         %__MODULE__{
           data: data,
           default_num_neighbors: default_num_neighbors,
           metric: metric
         } = _model,
         x
       ) do
    {num_samples, num_features} = Nx.shape(data)
    {num_samples_x, _num_features} = Nx.shape(x)
    broadcast_shape = {num_samples_x, num_samples, num_features}
    data_broadcast = Nx.new_axis(data, 0) |> Nx.broadcast(broadcast_shape)
    x_broadcast = Nx.new_axis(x, 1) |> Nx.broadcast(broadcast_shape)

    dist =
      case metric do
        {:minkowski, p} ->
          Scholar.Metrics.Distance.minkowski(
            data_broadcast,
            x_broadcast,
            axes: [-1],
            p: p
          )

        :cosine ->
          Scholar.Metrics.Distance.pairwise_cosine(x, data)
      end

    {val, ind} = Nx.top_k(-dist, k: default_num_neighbors)
    {-val, ind}
  end

  defnp check_weights(weights) do
    zero_mask = weights == 0
    zero_rows = zero_mask |> Nx.any(axes: [1], keep_axes: true) |> Nx.broadcast(weights)
    weights = Nx.select(zero_mask, 1, weights)
    weights_inv = 1 / weights
    Nx.select(zero_rows, Nx.select(zero_mask, 1, 0), weights_inv)
  end

  defnp mode_weighted(tensor, weights, opts \\ []) do
    axis = opts[:axis]
    tensor_size = Nx.size(tensor)

    cond do
      tensor_size == 1 ->
        Nx.squeeze(tensor, axes: [axis])

      true ->
        weighted_mode_general(tensor, weights, axis: axis)
    end
  end

  defnp weighted_mode_general(tensor, weights, opts) do
    axis = opts[:axis]

    {tensor, weights, axis} =
      if axis == 0,
        do: {Nx.transpose(tensor), Nx.transpose(weights), 1},
        else: {tensor, weights, axis}

    {num_samples, num_features} = tensor_shape = Nx.shape(tensor)

    indices = Nx.argsort(tensor, axis: axis)

    sorted = Nx.take_along_axis(tensor, indices, axis: axis)

    size_to_broadcast = {num_samples, 1}

    group_indices =
      Nx.concatenate(
        [
          Nx.broadcast(0, size_to_broadcast),
          Nx.not_equal(
            Nx.slice_along_axis(sorted, 0, Nx.axis_size(sorted, axis) - 1, axis: axis),
            Nx.slice_along_axis(sorted, 1, Nx.axis_size(sorted, axis) - 1, axis: axis)
          )
        ],
        axis: axis
      )
      |> Nx.cumulative_sum(axis: axis)

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
      |> Nx.argmax(axis: axis, keep_axis: true)

    indices =
      largest_group_indices
      |> Nx.broadcast(group_indices)
      |> Nx.equal(group_indices)
      |> Nx.argmax(axis: axis, keep_axis: true)

    res = Nx.take_along_axis(sorted, indices, axis: axis)
    Nx.squeeze(res, axes: [axis])
  end
end
