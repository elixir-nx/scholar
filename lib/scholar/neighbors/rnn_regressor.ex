defmodule Scholar.Neighbors.RadiusNNRegressor do
  @moduledoc """
  The Radius Nearest Neighbors.

  It implements regression.
  """
  import Nx.Defn

  @derive {Nx.Container,
           keep: [:weights, :num_classes, :metric, :radius], containers: [:data, :labels]}
  defstruct [:data, :labels, :weights, :num_classes, :metric, :radius]

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
      type: {:custom, Scholar.Neighbors.Utils, :pairwise_metric, []},
      default: &Scholar.Metrics.Distance.pairwise_minkowski/2,
      doc: ~S"""
      The function that measures the pairwise distance between two points. Possible values:

      * `{:minkowski, p}` - Minkowski metric. By changing the value of `p` parameter (a positive number or `:infinity`)
      we can set Manhattan (`1`), Euclidean (`2`), Chebyshev (`:infinity`), or any arbitrary $L_p$ metric.

      * `:cosine` - Cosine metric.

      * Anonymous function of arity 2 that takes two rank-2 tensors.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fit the Radius nearest neighbors classifier from the training data set.

  Currently 2D labels are only supported.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:data` - Training data.

    * `:labels` - Labels of each point.

    * `:weights` - Weight function used in prediction.

    * `:num_classes` - Number of classes in provided labels.

    * `:metric` - The metric function used.

    * `:radius` - Radius of neighborhood.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> Scholar.Neighbors.RadiusNNRegressor.fit(x, y, num_classes: 2)
      %Scholar.Neighbors.RadiusNNRegressor{
        data: Nx.tensor(
          [
            [1, 2],
            [2, 4],
            [1, 3],
            [2, 5]
          ]
        ),
        labels: Nx.tensor([1, 0, 1, 1]),
        weights: :uniform,
        num_classes: 2,
        metric: &Scholar.Metrics.Distance.pairwise_minkowski/2,
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

    %__MODULE__{
      data: x,
      labels: y,
      weights: opts[:weights],
      num_classes: opts[:num_classes],
      metric: opts[:metric],
      radius: opts[:radius]
    }
  end

  @doc """
  Makes predictions with the given `model` on inputs `x`.

  ## Return Values

  It returns a tensor with predicted class labels.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> model = Scholar.Neighbors.RadiusNNRegressor.fit(x, y, num_classes: 2)
      iex> Scholar.Neighbors.RadiusNNRegressor.predict(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      #Nx.Tensor<
        f32[2]
        [0.5, 1.0]
      >
  """
  defn predict(%__MODULE__{labels: labels, weights: weights} = model, x) do
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

  @doc """
  Find the Radius neighbors of a point.

  ## Return Values

  Returns indices of the selected neighbor points as a mask  (1 if a point is a neighbor, 0 otherwise) and their respective distances.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> model = Scholar.Neighbors.RadiusNNRegressor.fit(x, y, num_classes: 2)
      iex> {distances, mask} = Scholar.Neighbors.RadiusNNRegressor.radius_neighbors(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      iex> distances
      #Nx.Tensor<
        f32[2][4]
        [
          [2.469818353652954, 0.3162313997745514, 1.5811394453048706, 0.7071067690849304],
          [0.10000114142894745, 2.1931710243225098, 1.0049877166748047, 3.132091760635376]
        ]
      >
      iex> mask
      #Nx.Tensor<
        u8[2][4]
        [
          [0, 1, 0, 1],
          [1, 0, 0, 0]
        ]
      >
  """
  defn radius_neighbors(%__MODULE__{metric: metric, radius: radius, data: data}, x) do
    distances = metric.(x, data)
    {distances, distances <= radius}
  end

  defnp check_weights(weights) do
    zero_mask = weights == 0
    zero_rows = zero_mask |> Nx.any(axes: [1], keep_axes: true) |> Nx.broadcast(weights)
    weights = Nx.select(zero_mask, 1, weights)
    weights_inv = 1 / weights
    Nx.select(zero_rows, Nx.select(zero_mask, 1, 0), weights_inv)
  end
end
