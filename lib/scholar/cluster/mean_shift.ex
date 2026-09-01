defmodule Scholar.Cluster.MeanShift do
  @moduledoc """
  Mean shift clustering.

  Mean shift moves every seed towards the mean of the samples inside a ball of
  radius `:bandwidth` around it, repeating until the seed stops moving. Seeds
  that settle within one bandwidth of each other describe the same mode, so the
  weaker ones are discarded and the survivors become the cluster centers.

  The number of clusters is discovered from the data rather than given, and the
  bandwidth is what controls it.

  Distances are Euclidean. The update step averages the samples in a
  neighborhood, which only has meaning in a space where the arithmetic mean is
  the centroid.

  The time complexity is $O(I * S * N)$ for $N$ samples, $S$ seeds and $I$
  iterations. The space complexity is $O(S * N)$.
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:cluster_centers, :labels, :num_clusters, :iterations]}
  defstruct [:cluster_centers, :labels, :num_clusters, :iterations]

  opts = [
    bandwidth: [
      required: true,
      type: {:custom, Scholar.Options, :positive_number, []},
      doc: """
      The radius of the region a seed averages over. Larger values merge more
      modes together and yield fewer clusters.
      """
    ],
    max_iterations: [
      default: 300,
      type: :pos_integer,
      doc: """
      The maximum number of times every seed is moved. Seeds stop early once
      none of them moves further than `bandwidth / 1000`. Note that
      scikit-learn's `max_iter` checks the limit after moving, so it takes one
      more step than the number given.
      """
    ],
    cluster_all: [
      default: true,
      type: :boolean,
      doc: """
      If `true`, every sample is assigned to the nearest cluster center. If
      `false`, samples further than `:bandwidth` from every center are labeled
      `-1`.
      """
    ],
    seeds: [
      doc: """
      The points to start from, given as a tensor of shape `{num_seeds,
      num_features}`. Defaults to the samples themselves. Fewer seeds make the
      fit cheaper at the risk of missing a mode.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a mean shift model for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:cluster_centers` - The point every seed settled on, ordered by how many
      samples it gathered. Rows that lost to a stronger center are set to
      `:infinity`, so the tensor keeps the shape of the seeds. Use `prune/1` to
      drop them.

    * `:labels` - The row of `:cluster_centers` each sample belongs to, or `-1`
      when `:cluster_all` is `false` and the sample is further than
      `:bandwidth` from every center.

    * `:num_clusters` - The number of centers that survived.

    * `:iterations` - How many times the seeds were moved. This is at least one,
      since the seeds have to move once for their movement to be measured.

  ## Examples

      iex> x = Nx.tensor([[1, 1], [1.2, 1.1], [8, 8], [8.1, 8.2]])
      iex> Scholar.Cluster.MeanShift.fit(x, bandwidth: 1.0)
      %Scholar.Cluster.MeanShift{
        cluster_centers: Nx.f32(
          [
            [8.050000190734863, 8.100000381469727],
            [:infinity, :infinity],
            [1.100000023841858, 1.0499999523162842],
            [:infinity, :infinity]
          ]
        ),
        labels: Nx.s32([2, 2, 0, 0]),
        num_clusters: Nx.u32(2),
        iterations: Nx.u32(2)
      }
  """
  deftransform fit(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    {seeds, opts} = Keyword.pop(opts, :seeds, {})

    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {num_samples, num_features}, got: #{inspect(Nx.shape(x))}"
    end

    if Nx.rank(seeds) != 0 and
         (Nx.rank(seeds) != 2 or Nx.axis_size(seeds, 1) != Nx.axis_size(x, 1)) do
      raise ArgumentError,
            "expected seeds to have shape {num_seeds, #{Nx.axis_size(x, 1)}}, " <>
              "got: #{inspect(Nx.shape(seeds))}"
    end

    fit_n(x, seeds, opts)
  end

  defnp fit_n(x, seeds, opts) do
    x = to_float(x)

    seeds =
      case seeds do
        {} -> x
        _ -> to_float(seeds)
      end

    # the seeds carry the loop's accumulator, so they and the samples have to
    # agree on a type or the moved seeds come back wider than they went in
    type = Nx.Type.merge(Nx.type(x), Nx.type(seeds))
    x = Nx.as_type(x, type)
    seeds = Nx.as_type(seeds, type)

    bandwidth = opts[:bandwidth]
    {centers, iterations} = shift_seeds(x, seeds, bandwidth, opts)

    gathered = intensity(x, centers, bandwidth)
    order = strongest_first(gathered, centers)
    centers = Nx.take(centers, order)

    # a seed that never had a sample in reach describes no mode at all
    kept = drop_duplicate_modes(centers, bandwidth) and Nx.take(gathered, order) > 0
    num_clusters = Nx.sum(kept)

    distances =
      Nx.select(
        Nx.broadcast(Nx.new_axis(kept, 0), {Nx.axis_size(x, 0), Nx.axis_size(centers, 0)}),
        Scholar.Metrics.Distance.pairwise_euclidean(x, centers),
        Nx.Constants.infinity(to_float_type(x))
      )

    labels = Nx.argmin(distances, axis: 1) |> Nx.as_type(:s32)

    labels =
      if opts[:cluster_all] do
        labels
      else
        Nx.select(Nx.reduce_min(distances, axes: [1]) <= bandwidth, labels, -1)
      end

    labels = Nx.select(num_clusters > 0, labels, -1)

    %__MODULE__{
      cluster_centers:
        Nx.select(
          Nx.broadcast(Nx.new_axis(kept, 1), Nx.shape(centers)),
          centers,
          Nx.Constants.infinity(to_float_type(x))
        ),
      labels: labels,
      num_clusters: Nx.as_type(num_clusters, :u32),
      iterations: iterations
    }
  end

  # every seed moves at once, which is one pairwise matrix per iteration rather
  # than one per seed
  defnp shift_seeds(x, seeds, bandwidth, opts) do
    tolerance = bandwidth / 1000

    {seeds, _, iterations} =
      while {seeds, {x, bandwidth, tolerance, moving = Nx.u8(1)}, i = Nx.u32(0)},
            i < opts[:max_iterations] and moving do
        within = Scholar.Metrics.Distance.pairwise_euclidean(seeds, x) <= bandwidth
        members = Nx.as_type(within, Nx.type(seeds))
        count = Nx.sum(members, axes: [1], keep_axes: true)

        # a seed with an empty neighborhood would divide by zero, so it stays put
        moved =
          Nx.select(
            Nx.broadcast(count > 0, Nx.shape(seeds)),
            Nx.dot(members, x) / Nx.select(count > 0, count, 1),
            seeds
          )

        shift = Scholar.Metrics.Distance.euclidean(moved, seeds, axes: [1])
        {moved, {x, bandwidth, tolerance, Nx.any(shift > tolerance)}, i + 1}
      end

    {seeds, iterations}
  end

  defnp intensity(x, centers, bandwidth) do
    Scholar.Metrics.Distance.pairwise_euclidean(centers, x)
    |> Nx.less_equal(bandwidth)
    |> Nx.sum(axes: [1])
  end

  # ties break on the coordinates, descending, so that seeds landing on the same
  # mode keep a stable order
  deftransformp strongest_first(intensity, centers) do
    keys =
      Enum.map((Nx.axis_size(centers, 1) - 1)..0//-1, &centers[[.., &1]]) ++ [intensity]

    Enum.reduce(keys, Nx.iota({Nx.axis_size(centers, 0)}, type: :s32), fn key, order ->
      Nx.take(order, Nx.argsort(Nx.take(key, order), direction: :desc, stable: true))
    end)
  end

  defnp drop_duplicate_modes(centers, bandwidth) do
    num_centers = Nx.axis_size(centers, 0)

    {kept, _} =
      while {kept = Nx.broadcast(Nx.u8(1), {num_centers}), {centers, bandwidth, i = Nx.u32(0)}},
            i < num_centers do
        kept =
          if kept[i] do
            near =
              Scholar.Metrics.Distance.euclidean(centers, Nx.new_axis(centers[i], 0), axes: [1]) <=
                bandwidth

            Nx.indexed_put(Nx.select(near, Nx.u8(0), kept), Nx.new_axis(i, 0), Nx.u8(1))
          else
            kept
          end

        {kept, {centers, bandwidth, i + 1}}
      end

    kept
  end

  @doc """
  Drops the centers that lost to a stronger one and renumbers the labels.

  `fit/2` keeps one row per seed so the shapes stay static. This returns a model
  holding only the `:num_clusters` centers that survived, with `:labels`
  renumbered to index them.

  ## Examples

      iex> x = Nx.tensor([[1, 1], [1.2, 1.1], [8, 8], [8.1, 8.2]])
      iex> model = Scholar.Cluster.MeanShift.fit(x, bandwidth: 1.0)
      iex> Scholar.Cluster.MeanShift.prune(model)
      %Scholar.Cluster.MeanShift{
        cluster_centers: Nx.f32(
          [
            [8.050000190734863, 8.100000381469727],
            [1.100000023841858, 1.0499999523162842]
          ]
        ),
        labels: Nx.s32([1, 1, 0, 0]),
        num_clusters: Nx.u32(2),
        iterations: Nx.u32(2)
      }
  """
  def prune(%__MODULE__{cluster_centers: centers, labels: labels} = model) do
    if Nx.to_number(model.num_clusters) == 0 do
      raise ArgumentError,
            "the model has no clusters to keep, every seed was further than the bandwidth " <>
              "from all samples"
    end

    kept =
      centers
      |> Nx.is_infinity()
      |> Nx.all(axes: [1])
      |> Nx.equal(0)

    indices = kept |> Nx.to_flat_list() |> Enum.with_index() |> Enum.filter(&(elem(&1, 0) == 1))
    indices = Enum.map(indices, &elem(&1, 1))

    renumber =
      indices
      |> Enum.with_index()
      |> Map.new()
      |> then(fn mapping -> fn old -> Map.get(mapping, old, -1) end end)

    %__MODULE__{
      model
      | cluster_centers: Nx.take(centers, Nx.tensor(indices)),
        labels:
          labels
          |> Nx.to_flat_list()
          |> Enum.map(renumber)
          |> Nx.tensor(type: Nx.type(labels))
    }
  end
end
