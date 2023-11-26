defmodule Scholar.Neighbors.RadiusNearestNeighbors do
  @moduledoc """
  The Radius Nearest Neighbors.

  It implements both classification and regression.
  """
  import Nx.Defn
  import Scholar.Shared
  require Nx

  @derive {Nx.Container,
           keep: [:weights, :num_classes, :task, :metric, :radius], containers: [:data, :labels]}
  defstruct [:data, :labels, :weights, :num_classes, :task, :metric, :radius]

  opts = [
    radius: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0,
      doc: "Radius of neighborhood"
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
      Task that will be performed using Radius Nearest Neighbors. Possible values:

      * `:classification` - Classifier implementing the Radius Nearest Neighbors vote.

      * `:regression` - Regression based on Radius Nearest Neighbors.
        The target is predicted by local interpolation of the targets associated of the nearest neighbors in the training set.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fit the Radius nearest neighbors classifier from the training data set.

  For classification, provided labels need to be consecutive non-negative integers. If your labels does
  not meet this condition please use `Scholar.Preprocessing.ordinal_encode`

  Currently 2D labels are only supported for regression tasks.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:data` - Training data.

    * `:labels` - Labels of each point.

    * `:weights` - Weight function used in prediction.

    * `:num_classes` - Number of classes in provided labels.

    * `:task` - Task that will be performed using Radius Nearest Neighbors.
    For `:classification` task, model will be a classifier implementing the Radius Nearest Neighbors vote.
    For `:regression` task, model is a regressor based on Radius Nearest Neighbors.

    * `:metric` - Name of the metric.

    * `:radius` - Radius of neighborhood.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> Scholar.Neighbors.RadiusNearestNeighbors.fit(x, y, num_classes: 2)
      %Scholar.Neighbors.RadiusNearestNeighbors{
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
        weights: :uniform,
        num_classes: 2,
        task: :classification,
        metric: {:minkowski, 2},
        radius: 1.0
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
      weights: opts[:weights],
      num_classes: opts[:num_classes],
      task: opts[:task],
      metric: opts[:metric],
      radius: opts[:radius]
    }
  end

  @doc """
  Makes predictions with the given `model` on inputs `x`.

  ## Return Values

    It returns a tensor with predicted class labels

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> model = Scholar.Neighbors.RadiusNearestNeighbors.fit(x, y, num_classes: 2)
      iex> Scholar.Neighbors.RadiusNearestNeighbors.predict(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      Nx.tensor(
        [0, 1]
      )
  """
  defn predict(%__MODULE__{labels: labels, weights: weights, task: task} = model, x) do
    case task do
      :classification ->
        {probabilities, outliers_mask} = predict_proba(model, x)
        results = Nx.argmax(probabilities, axis: 1)
        Nx.select(outliers_mask, -1, results)

      :regression ->
        {distances, indices} = radius_neighbors(model, x)

        x_num_samples = Nx.axis_size(x, 0)
        train_num_samples = Nx.axis_size(labels, 0)
        labels_rank = Nx.rank(labels)

        labels =
          if labels_rank == 1 do
            Nx.new_axis(labels, 0) |> Nx.broadcast({x_num_samples, train_num_samples})
          else
            out_size = Nx.axis_size(labels, 1)
            Nx.new_axis(labels, 0) |> Nx.broadcast({x_num_samples, train_num_samples, out_size})
          end

        indices =
          if labels_rank == 2,
            do: Nx.new_axis(indices, -1) |> Nx.broadcast(labels),
            else: indices

        case weights do
          :distance ->
            weights = check_weights(distances)

            weights =
              if labels_rank == 2,
                do: Nx.new_axis(weights, -1) |> Nx.broadcast(labels),
                else: weights

            Nx.weighted_mean(labels, indices * weights, axes: [1])

          :uniform ->
            Nx.weighted_mean(labels, indices, axes: [1])
        end
    end
  end

  @doc """
  Return probability estimates for the test data `x`.

  ## Return Values

    It returns a typle with tensor with probabilities of classes and mask of outliers.
    They are arranged in lexicographic order.


  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> model = Scholar.Neighbors.RadiusNearestNeighbors.fit(x, y, num_classes: 2)
      iex> Scholar.Neighbors.RadiusNearestNeighbors.predict_proba(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      {Nx.tensor(
        [
          [0.5, 0.5],
          [0.0, 1.0]
        ]
      ),
      Nx.tensor(
        [0, 0], type: :u8
      )}
  """
  deftransform predict_proba(
                 %__MODULE__{
                   task: :classification
                 } = model,
                 x
               ) do
    predict_proba_n(model, x)
  end

  @doc """
  Find the Radius neighbors of a point.

  ## Return Values

    Returns indices of the selected neighbor points as a mask  (1 if a point is a neighbor, 0 otherwise) and their respective distances.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> model = Scholar.Neighbors.RadiusNearestNeighbors.fit(x, y, num_classes: 2)
      iex> Scholar.Neighbors.RadiusNearestNeighbors.radius_neighbors(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      {Nx.tensor(
        [
          [2.469817876815796, 0.3162279427051544, 1.5811389684677124, 0.7071065902709961],
          [0.10000002384185791, 2.193171262741089, 1.0049875974655151, 3.132091760635376]
        ]
      ),
      Nx.tensor(
        [
          [0, 1, 0, 1],
          [1, 0, 0, 0]
        ], type: :u8
      )}
  """
  defn radius_neighbors(%__MODULE__{metric: metric, radius: radius, data: data}, x) do
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

    {dist, dist <= radius}
  end

  defnp predict_proba_n(
          %__MODULE__{
            labels: labels,
            weights: weights,
            num_classes: num_classes
          } = model,
          x
        ) do
    {distances, indices} = radius_neighbors(model, x)
    num_samples = Nx.axis_size(x, 0)
    outliers_mask = Nx.sum(indices, axes: [1]) == 0

    probabilities =
      Nx.broadcast(Nx.tensor(0.0, type: to_float_type(x)), {num_samples, num_classes})

    weights_vals =
      case weights do
        :distance -> check_weights(distances)
        :uniform -> Nx.broadcast(Nx.tensor(1.0, type: to_float_type(x)), indices)
      end

    {final_probabilities, _} =
      while {probabilities, {labels, weights_vals, indices, i = 0}}, i < num_classes do
        class_mask = (labels == i) |> Nx.new_axis(0)
        class_sum = (indices * class_mask * weights_vals) |> Nx.sum(axes: [1], keep_axes: true)
        probabilities = Nx.put_slice(probabilities, [0, i], class_sum)
        {probabilities, {labels, weights_vals, indices, i + 1}}
      end

    normalizer = Nx.sum(final_probabilities, axes: [1])
    normalizer = Nx.select(normalizer == 0, 1, normalizer)
    {final_probabilities / Nx.new_axis(normalizer, -1), outliers_mask}
  end

  defnp check_weights(weights) do
    zero_mask = weights == 0
    zero_rows = zero_mask |> Nx.any(axes: [1], keep_axes: true) |> Nx.broadcast(weights)
    weights = Nx.select(zero_mask, 1, weights)
    weights_inv = 1 / weights
    Nx.select(zero_rows, Nx.select(zero_mask, 1, 0), weights_inv)
  end
end
