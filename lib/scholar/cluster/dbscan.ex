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
      type: {:custom, Scholar.Options, :metric, []},
      default: {:minkowski, 2},
      doc: ~S"""
      Name of the metric. Possible values:

      * `{:minkowski, p}` - Minkowski metric. By changing value of `p` parameter (a positive number or `:infinity`)
        we can set Manhattan (`1`), Euclidean (`2`), Chebyshev (`:infinity`), or any arbitrary $L_p$ metric.

      * `:cosine` - Cosine metric.
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
      Scholar.Neighbors.RadiusNearestNeighbors.fit(x, y_dummy,
        num_classes: 1,
        radius: opts[:eps],
        metric: opts[:metric]
      )

    {_dist, indices} =
      Scholar.Neighbors.RadiusNearestNeighbors.radius_neighbors(neighbor_model, x)

    n_neigbors = Nx.sum(indices * weights, axes: [1])
    core_samples = n_neigbors >= opts[:min_samples]
    labels = dbscan_inner(core_samples, indices)

    %__MODULE__{
      core_sample_indices: core_samples,
      labels: labels
    }
  end

  defnp dbscan_inner(is_core?, indices) do
    {labels, _} =
      while {labels = Nx.broadcast(0, {Nx.axis_size(indices, 0)}),
             {indices, is_core?, label_num = 1, i = 0}},
            i < Nx.axis_size(indices, 0) do
        stack = Nx.broadcast(0, {Nx.axis_size(indices, 0) ** 2})
        stack_ptr = 0

        if Nx.take(labels, i) != 0 or not Nx.take(is_core?, i) do
          {labels, {indices, is_core?, label_num, i + 1}}
        else
          {labels, _} =
            while {labels, {k = i, label_num, indices, is_core?, stack, stack_ptr}},
                  stack_ptr >= 0 do
              {labels, stack, stack_ptr} =
                if Nx.take(labels, k) == 0 do
                  labels =
                    Nx.indexed_put(
                      labels,
                      Nx.new_axis(Nx.new_axis(k, 0), 0),
                      Nx.new_axis(label_num, 0)
                    )

                  {stack, stack_ptr} =
                    if Nx.take(is_core?, k) do
                      neighb = Nx.take(indices, k)
                      mask = neighb * (labels == 0)

                      {stack, stack_ptr, _} =
                        while {stack, stack_ptr, {mask, j = 0}}, j < Nx.axis_size(mask, 0) do
                          if Nx.take(mask, j) != 0 do
                            stack =
                              Nx.indexed_put(
                                stack,
                                Nx.new_axis(Nx.new_axis(stack_ptr, 0), 0),
                                Nx.new_axis(j, 0)
                              )

                            {stack, stack_ptr + 1, {mask, j + 1}}
                          else
                            {stack, stack_ptr, {mask, j + 1}}
                          end
                        end

                      {stack, stack_ptr}
                    else
                      {stack, stack_ptr}
                    end

                  {labels, stack, stack_ptr}
                else
                  {labels, stack, stack_ptr}
                end

              k = if stack_ptr > 0, do: Nx.take(stack, stack_ptr - 1), else: -1
              stack_ptr = stack_ptr - 1
              {labels, {k, label_num, indices, is_core?, stack, stack_ptr}}
            end

          {labels, {indices, is_core?, label_num + 1, i + 1}}
        end
      end

    # we need to subtract 1 from labels because we started from label_num=1 which simplifies oprations
    labels - 1
  end
end
