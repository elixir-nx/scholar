defmodule Scholar.Cluster.AffinityPropagation do
  @moduledoc """
  Model representing affinity propagation clustering. The first dimension
  of `:clusters_centers` is set to the number of samples in the dataset.
  The artificial centers are filled with `:infinity` values. To fillter
  them out use `prune` function.

  The algorithm has a time complexity of the order $O(N^2T)$, where $N$ is
  the number of samples and $T$ is the number of iterations until convergence.
  Further, the memory complexity is of the order $O(N^2)$.
  """

  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container,
           containers: [
             :labels,
             :cluster_centers_indices,
             :cluster_centers,
             :num_clusters,
             :iterations
           ]}
  defstruct [
    :labels,
    :cluster_centers_indices,
    :cluster_centers,
    :num_clusters,
    :iterations
  ]

  @opts_schema [
    iterations: [
      type: :pos_integer,
      default: 300,
      doc: "Number of iterations of the algorithm."
    ],
    damping_factor: [
      type: :float,
      default: 0.5,
      doc: """
      Damping factor in the range [0.5, 1.0) is the extent to which the
      current value is maintained relative to incoming values (weighted 1 - damping).
      """
    ],
    preference: [
      type: {:or, [:float, :atom]},
      default: :reduce_min,
      doc: """
      How to compute the preferences for each point - points with larger values
      of preferences are more likely to be chosen as exemplars. The number of clusters is
      influenced by this option.

      The preferences is either an atom, each is a `Nx` reduction function to
      apply on the input similarities (such as `:reduce_min`, `:median`, `:mean`,
      etc) or a float.
      """
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Determines random number generation for centroid initialization.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ],
    learning_loop_unroll: [
      type: :boolean,
      default: false,
      doc: ~S"""
      If `true`, the learning loop is unrolled.
      """
    ],
    converge_after: [
      type: :pos_integer,
      default: 15,
      doc: ~S"""
      Number of iterations with no change in the number of estimated clusters
      that stops the convergence.
      """
    ]
  ]

  @doc """
  Cluster the dataset using affinity propagation.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:clusters_centers` - Cluster centers from the initial data.

    * `:cluster_centers_indices` - Indices of cluster centers.

    * `:num_clusters` - Number of clusters.

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> x = Nx.tensor([[12,5,78,2], [9,3,81,-2], [-1,3,6,1], [1,-2,5,2]])
      iex> Scholar.Cluster.AffinityPropagation.fit(x, key: key)
      %Scholar.Cluster.AffinityPropagation{
        labels: Nx.tensor([0, 0, 2, 2]),
        cluster_centers_indices: Nx.tensor([0, -1, 2, -1]),
        cluster_centers: Nx.tensor(
          [
            [12.0, 5.0, 78.0, 2.0],
            [:infinity, :infinity, :infinity, :infinity],
            [-1.0, 3.0, 6.0, 1.0],
            [:infinity, :infinity, :infinity, :infinity]
          ]
        ),
        num_clusters: Nx.tensor(2, type: :u64),
        iterations: Nx.tensor(22, type: :s64)
      }
  """
  deftransform fit(data, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_n(data, key, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(data, key, opts) do
    data = to_float(data)
    iterations = opts[:iterations]
    damping_factor = opts[:damping_factor]
    converge_after = opts[:converge_after]
    n = Nx.axis_size(data, 0)
    s = initialize_similarity(data, opts)

    zero_n = Nx.tensor(0, type: Nx.type(s)) |> Nx.broadcast({n, n})
    {normal, _new_key} = Nx.Random.normal(key, 0, 1, shape: {n, n}, type: Nx.type(s))

    s =
      s +
        normal *
          (Nx.Constants.smallest_positive_normal(Nx.type(s)) * 100 +
             Nx.Constants.epsilon(Nx.type(data)) / 10 * s)

    range = Nx.iota({n})

    e = Nx.broadcast(Nx.s64(0), {n, converge_after})
    stop = Nx.u8(0)

    {{a, r, it}, _} =
      while {{a = zero_n, r = zero_n, i = 0}, {s, range, stop, e}},
            i < iterations and not stop do
        temp = a + s
        indices = Nx.argmax(temp, axis: 1)
        y = Nx.reduce_max(temp, axes: [1])

        neg_inf = Nx.Constants.neg_infinity(to_float_type(a))
        neg_infinities = Nx.broadcast(neg_inf, {n})
        max_indices = Nx.stack([range, indices], axis: 1)
        temp = Nx.indexed_put(temp, max_indices, neg_infinities)
        y2 = Nx.reduce_max(temp, axes: [1])

        temp = s - Nx.new_axis(y, -1)
        temp = Nx.indexed_put(temp, max_indices, Nx.gather(s, max_indices) - y2)
        temp = temp * (1 - damping_factor)
        r = r * damping_factor + temp

        temp = Nx.max(r, 0)
        temp = Nx.put_diagonal(temp, Nx.take_diagonal(r))
        temp = temp - Nx.sum(temp, axes: [0])
        a_change = Nx.take_diagonal(temp)

        temp = Nx.max(temp, 0)
        temp = Nx.put_diagonal(temp, a_change)
        temp = temp * (1 - damping_factor)
        a = a * damping_factor - temp

        curr_e = Nx.take_diagonal(a) + Nx.take_diagonal(r) > 0
        curr_e_slice = Nx.reshape(curr_e, {:auto, 1})
        e = Nx.put_slice(e, [0, Nx.remainder(i, converge_after)], curr_e_slice)
        k = Nx.sum(curr_e, axes: [0])

        stop =
          if i >= converge_after do
            se = Nx.sum(e, axes: [1])
            unconverged = Nx.sum((se == 0) + (se == converge_after)) != n

            if (not unconverged and k > 0) or i == iterations do
              Nx.u8(1)
            else
              stop
            end
          end

        {{a, r, i + 1}, {s, range, stop, e}}
      end

    diagonals = Nx.take_diagonal(a) + Nx.take_diagonal(r) > 0

    k = Nx.sum(diagonals, axes: [0])
    {n, _} = shape = Nx.shape(data)

    {cluster_centers, cluster_centers_indices, labels} =
      if k > 0 do
        mask = diagonals != 0

        indices =
          Nx.select(mask, Nx.iota(Nx.shape(diagonals)), -1)
          |> Nx.as_type({:s, 64})

        cluster_centers =
          Nx.select(
            Nx.broadcast(Nx.new_axis(mask, -1), shape),
            data,
            Nx.Constants.infinity(to_float_type(a))
          )

        labels =
          Nx.broadcast(mask, Nx.shape(s))
          |> Nx.select(s, Nx.Constants.neg_infinity(Nx.type(s)))
          |> Nx.argmax(axis: 1)
          |> Nx.as_type({:s, 64})

        labels = Nx.select(mask, Nx.iota(Nx.shape(labels)), labels)

        {cluster_centers, indices, labels}
      else
        {Nx.tensor(-1, type: Nx.type(data)), Nx.broadcast(Nx.tensor(-1, type: :s64), {n}),
         Nx.broadcast(Nx.tensor(-1, type: :s64), {n})}
      end

    %__MODULE__{
      cluster_centers_indices: cluster_centers_indices,
      cluster_centers: cluster_centers,
      labels: labels,
      num_clusters: k,
      iterations: it
    }
  end

  defnp initialize_similarity(data, opts \\ []) do
    n = Nx.axis_size(data, 0)
    dist = -Scholar.Metrics.Distance.pairwise_squared_euclidean(data)
    preference = initialize_preference(dist, opts[:preference])
    Nx.put_diagonal(dist, Nx.broadcast(preference, {n}))
  end

  deftransformp initialize_preference(dist, preference) do
    cond do
      is_atom(preference) -> apply(Nx, preference, [dist])
      is_float(preference) -> preference
    end
  end

  @doc """
  Optionally prune clusters, indices, and labels to only valid entries.

  It returns an updated and pruned model.

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> x = Nx.tensor([[12,5,78,2], [9,3,81,-2], [-1,3,6,1], [1,-2,5,2]])
      iex> model = Scholar.Cluster.AffinityPropagation.fit(x, key: key)
      iex> Scholar.Cluster.AffinityPropagation.prune(model)
      %Scholar.Cluster.AffinityPropagation{
        labels: Nx.tensor([0, 0, 1, 1]),
        cluster_centers_indices: Nx.tensor([0, 2]),
        cluster_centers: Nx.tensor(
          [
            [12.0, 5.0, 78.0, 2.0],
            [-1.0, 3.0, 6.0, 1.0]
          ]
        ),
        num_clusters: Nx.tensor(2, type: :u64),
        iterations: Nx.tensor(22, type: :s64)
      }
  """
  def prune(
        %__MODULE__{
          cluster_centers_indices: cluster_centers_indices,
          cluster_centers: cluster_centers,
          labels: labels
        } = model
      ) do
    {indices, _, _, mapping} =
      cluster_centers_indices
      |> Nx.to_flat_list()
      |> Enum.reduce({[], 0, 0, []}, fn
        index, {indices, old_pos, new_pos, mapping} when index >= 0 ->
          {[index | indices], old_pos + 1, new_pos + 1, [{old_pos, new_pos} | mapping]}

        _index, {indices, old_pos, new_pos, mapping} ->
          {indices, old_pos + 1, new_pos, mapping}
      end)

    mapping = Map.new(mapping)
    cluster_centers_indices = Nx.tensor(Enum.reverse(indices))

    %__MODULE__{
      model
      | cluster_centers_indices: cluster_centers_indices,
        cluster_centers: Nx.take(cluster_centers, cluster_centers_indices),
        labels: labels |> Nx.to_flat_list() |> Enum.map(&Map.fetch!(mapping, &1)) |> Nx.tensor()
    }
  end

  @doc """
  Predict the closest cluster each sample in `x` belongs to.

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> x = Nx.tensor([[12,5,78,2], [9,3,81,-2], [-1,3,6,1], [1,-2,5,2]])
      iex> model = Scholar.Cluster.AffinityPropagation.fit(x, key: key)
      iex> model = Scholar.Cluster.AffinityPropagation.prune(model)
      iex> Scholar.Cluster.AffinityPropagation.predict(model, Nx.tensor([[10,3,50,6], [8,3,8,2]]))
      #Nx.Tensor<
        s64[2]
        [0, 1]
      >
  """
  defn predict(%__MODULE__{cluster_centers: cluster_centers} = _model, x) do
    dist = Scholar.Metrics.Distance.pairwise_euclidean(x, cluster_centers)

    Nx.select(Nx.is_nan(dist), Nx.Constants.infinity(Nx.type(dist)), dist)
    |> Nx.argmin(axis: 1)
  end
end
