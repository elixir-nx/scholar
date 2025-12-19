defmodule Scholar.Cluster.DBSCAN do
  @moduledoc """
  Perform DBSCAN clustering from vector array or distance matrix.

  DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
  Finds core samples of high density and expands clusters from them.
  Good for data which contains clusters of similar density.

  The time complexity is $O(N^2)$ for $N$ samples.
  The space complexity is $O(N^2)$.
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:core_sample_indices, :labels]}
  defstruct [:core_sample_indices, :labels]

  opts = [
    eps: [
      default: 0.5,
      doc: """
      The maximum distance between two samples for them to be considered as in the same neighborhood.
      """,
      type: {:custom, Scholar.Options, :positive_number, []}
    ],
    min_samples: [
      default: 5,
      doc: """
      The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
      This includes the point itself.
      """,
      type: :integer
    ],
    metric: [
      type: {:custom, Scholar.Neighbors.Utils, :pairwise_metric, []},
      default: &Scholar.Metrics.Distance.pairwise_minkowski/2,
      doc: ~S"""
      The function that measures the pairwise distance between two points. Possible values:

      * `{:minkowski, p}` - Minkowski metric. By changing value of `p` parameter (a positive number or `:infinity`)
      we can set Manhattan (`1`), Euclidean (`2`), Chebyshev (`:infinity`), or any arbitrary $L_p$ metric.

      * `:cosine` - Cosine metric.

      * Anonymous function of arity 2 that takes two rank-2 tensors.
      """
    ],
    weights: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: """
      The weights for each observation in `x`. If equals to `nil`,
      all observations are assigned equal weight.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Perform DBSCAN clustering from vector array or distance matrix.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

  * `:core_sample_indices` - Indices of core samples represented as a mask.
    The mask is a boolean array of shape `{num_samples}` where `1` indicates
    that the corresponding sample is a core sample and `0` otherwise.

  * `:labels` - Cluster labels for each point in the dataset given to fit().
    Noisy samples are given the label `-1`.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
      iex> Scholar.Cluster.DBSCAN.fit(x, eps: 3, min_samples: 2)
      %Scholar.Cluster.DBSCAN{
        core_sample_indices: Nx.tensor(
          [1, 1, 1, 1, 1, 0], type: :u8
        ),
        labels: Nx.tensor(
          [0, 0, 0, 1, 1, -1]
        )
      }
  """
  deftransform fit(x, opts \\ []) do
    fit_n(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(x, opts) do
    num_samples = Nx.axis_size(x, 0)
    weights = validate_weights(opts[:weights], num_samples, type: to_float_type(x))
    weights = if Nx.rank(weights) == 0, do: weights, else: Nx.new_axis(weights, -1)
    y_dummy = Nx.broadcast(Nx.tensor(0), {num_samples})

    neighbor_model =
      Scholar.Neighbors.RadiusNNClassifier.fit(x, y_dummy,
        num_classes: 1,
        radius: opts[:eps],
        metric: opts[:metric]
      )

    {_dist, indices} =
      Scholar.Neighbors.RadiusNNClassifier.radius_neighbors(neighbor_model, x)

    n_neighbors = Nx.sum(indices * weights, axes: [1])
    core_samples = n_neighbors >= opts[:min_samples]
    labels = dbscan_inner(core_samples, indices)

    %__MODULE__{
      core_sample_indices: core_samples,
      labels: labels
    }
  end

  defnp dbscan_inner(is_core?, neighbors) do
    # We implement the clustering via label propagation.
    #
    # Algorithm:
    #
    # 1. Initialize each core sample with a unique label (its index),
    #    non-core samples get a "dummy label", which is bigger than
    #    all others.
    # 2. Then iteratively, we update each sample with minimum label
    #    from all of its core neighbors.
    # 3. Connected core samples (and all their neighbors) converge
    #    to the same minimum label.
    # 4. Isolated non-core samples are left with "dummy label", which
    #    we map to -1 at the end.
    #
    # This converges in O(D) iterations where D is the diameter of
    # the largest cluster.
    #
    # This approach is more parallelization-friendly than a sequential
    # sample-by-sample DFS traversal.

    num_samples = Nx.axis_size(is_core?, 0)
    dummy_label = num_samples

    labels = Nx.select(is_core?, Nx.iota({num_samples}), dummy_label)

    core_neighbors = Nx.new_axis(is_core?, 0) and neighbors

    # We create a tensor where for each sample (0-axis) we have indices
    # of its core neighbors (1-axis) and remaining spots filled with
    # its own index.
    core_neighbor_indices =
      Nx.select(
        core_neighbors,
        # neighbor index
        Nx.iota({num_samples, num_samples}, axis: 1),
        # self index
        Nx.iota({num_samples, num_samples}, axis: 0)
      )

    {labels, _, _} =
      while {labels, core_neighbor_indices, finished? = Nx.tensor(false)}, not finished? do
        core_neighbor_labels = Nx.take(labels, core_neighbor_indices)
        updated_labels = Nx.reduce_min(core_neighbor_labels, axes: [1])
        finished? = Nx.all(labels == updated_labels)
        {updated_labels, core_neighbor_indices, finished?}
      end

    # Normalize labels to be consecutive.
    normalized_labels = normalize_labels(labels)

    # Noisy samples don't get any label from core samples, so they keep
    # the dummy label, which we replace with -1.
    Nx.select(labels == dummy_label, -1, normalized_labels)
  end

  # Normalizes non-consecutive labels into consecutive labels.
  #
  # For example [1, 4, 2, 2, 1, 4] -> [0, 2, 1, 1, 0, 2].
  defnp normalize_labels(labels) do
    sort_indices = Nx.argsort(labels)
    unsort_indices = inverse_permutation(sort_indices)

    sorted = Nx.take_along_axis(labels, sort_indices)

    # Create a mask with 1 at every position where a new value appears,
    # then use cumulative sum, so that each group gets the same value.
    normalized_sorted =
      Nx.concatenate([
        Nx.tensor([0]),
        Nx.not_equal(sorted[0..-2//1], sorted[1..-1//1])
      ])
      |> Nx.cumulative_sum()

    Nx.take_along_axis(normalized_sorted, unsort_indices)
  end

  defnp inverse_permutation(indices) do
    shape = Nx.shape(indices)
    type = Nx.type(indices)

    Nx.indexed_put(
      Nx.broadcast(Nx.tensor(0, type: type), shape),
      Nx.new_axis(indices, -1),
      Nx.iota(shape, type: type)
    )
  end
end
