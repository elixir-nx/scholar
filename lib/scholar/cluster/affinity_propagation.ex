defmodule Scholar.Cluster.AffinityPropagation do
  @moduledoc """
  Model representing affinity propagation clustering.
  """

  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container,
           containers: [
             :labels,
             :cluster_centers_indices,
             :affinity_matrix,
             :cluster_centers,
             :num_clusters
           ]}
  defstruct [
    :labels,
    :cluster_centers_indices,
    :affinity_matrix,
    :cluster_centers,
    :num_clusters
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
    self_preference: [
      type: {:or, [:float, :boolean, :integer]},
      doc: "Self preference."
    ],
    seed: [
      type: :integer,
      doc: """
      Determines random number generation for centroid initialization.
      If the seed is not provided, it is set to `System.system_time()`.
      """
    ]
  ]

  @doc """
  Cluster the dataset using affinity propagation.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:affinity_matrix` - Affinity matrix. It is a negated squared euclidean distance of each pair of points.

    * `:clusters_centers` - Cluster centers from the initial data.

    * `:cluster_centers_indices` - Indices of cluster centers.

    * `:num_clusters` - Number of clusters.

  ## Examples

      iex> seed = 42
      iex> x = Nx.tensor([[12,5,78,2], [1,-5,7,32], [-1,3,6,1], [1,-2,5,2]])
      iex> Scholar.Cluster.AffinityPropagation.fit(x, seed: seed)
      %Scholar.Cluster.AffinityPropagation{
        labels: Nx.tensor([0, 3, 3, 3]),
        cluster_centers_indices: Nx.tensor([0, -1, -1, 3]),
        affinity_matrix: Nx.tensor(
          [
            [-0.0, -6162.0, -5358.0, -5499.0],
            [-6162.0, -0.0, -1030.0, -913.0],
            [-5358.0, -1030.0, -0.0, -31.0],
            [-5499.0, -913.0, -31.0, -0.0]
          ]),
        cluster_centers: Nx.tensor(
          [
            [12.0, 5.0, 78.0, 2.0],
            [:infinity, :infinity, :infinity, :infinity],
            [:infinity, :infinity, :infinity, :infinity],
            [1.0, -2.0, 5.0, 2.0]
          ]
        ),
        num_clusters: Nx.tensor(2, type: :u64)
      }
  """
  deftransform fit(data, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    opts = Keyword.update(opts, :self_preference, false, fn x -> x end)
    seed = Keyword.get_lazy(opts, :seed, &System.system_time/0)
    fit_n(data, seed, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(data, seed, opts) do
    data = to_float(data)
    iterations = opts[:iterations]
    damping_factor = opts[:damping_factor]
    self_preference = opts[:self_preference]

    {initial_a, initial_r, s, affinity_matrix} =
      initialize_matrices(data, self_preference: self_preference)

    {n, _} = Nx.shape(initial_a)
    key = Nx.Random.key(seed)
    {normal, _new_key} = Nx.Random.normal(key, 0, 1, shape: {n, n}, type: Nx.type(s))

    s =
      s +
        normal *
          (Nx.Constants.smallest_positive_normal(Nx.type(s)) * 100 +
             Nx.Constants.epsilon(Nx.type(data)) / 10 * s)

    range = Nx.iota({n})

    {a, r, _, _, _} =
      while {a = initial_a, r = initial_r, s, range, i = 0},
            i < iterations do
        temp = a + s
        indices = Nx.argmax(temp, axis: 1)
        y = Nx.reduce_max(temp, axes: [1])

        neg_inf = Nx.Constants.neg_infinity(Nx.Type.to_floating(Nx.type(a)))
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

        {a, r, s, range, i + 1}
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
            Nx.Constants.infinity(Nx.Type.to_floating(Nx.type(a)))
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
      affinity_matrix: affinity_matrix,
      cluster_centers_indices: cluster_centers_indices,
      cluster_centers: cluster_centers,
      labels: labels,
      num_clusters: k
    }
  end

  @doc """
  Optionally prune clusters, indices, and labels to only valid entries.

  It returns an updated and pruned model.

  ## Examples

      iex> seed = 42
      iex> x = Nx.tensor([[12,5,78,2], [1,-5,7,32], [-1,3,6,1], [1,-2,5,2]])
      iex> model = Scholar.Cluster.AffinityPropagation.fit(x, seed: seed)
      iex> Scholar.Cluster.AffinityPropagation.prune(model)
      %Scholar.Cluster.AffinityPropagation{
        labels: Nx.tensor([0, 1, 1, 1]),
        cluster_centers_indices: Nx.tensor([0, 3]),
        affinity_matrix: Nx.tensor(
          [
            [-0.0, -6162.0, -5358.0, -5499.0],
            [-6162.0, -0.0, -1030.0, -913.0],
            [-5358.0, -1030.0, -0.0, -31.0],
            [-5499.0, -913.0, -31.0, -0.0]
          ]),
        cluster_centers: Nx.tensor(
          [
            [12.0, 5.0, 78.0, 2.0],
            [1.0, -2.0, 5.0, 2.0]
          ]
        ),
        num_clusters: Nx.tensor(2, type: :u64)
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

      iex> seed = 42
      iex> x = Nx.tensor([[12,5,78,2], [1,5,7,32], [1,3,6,1], [1,2,5,2]])
      iex> model = Scholar.Cluster.AffinityPropagation.fit(x, seed: seed)
      iex> model = Scholar.Cluster.AffinityPropagation.prune(model)
      iex> Scholar.Cluster.AffinityPropagation.predict(model, Nx.tensor([[1,6,2,6], [8,3,8,2]]))
      #Nx.Tensor<
        s64[2]
        [1, 1]
      >
  """
  defn predict(%__MODULE__{cluster_centers: cluster_centers} = _model, x) do
    {num_clusters, num_features} = Nx.shape(cluster_centers)
    {num_samples, _} = Nx.shape(x)
    broadcast_shape = {num_samples, num_clusters, num_features}

    Scholar.Metrics.Distance.euclidean(
      Nx.new_axis(x, 1) |> Nx.broadcast(broadcast_shape),
      Nx.new_axis(cluster_centers, 0) |> Nx.broadcast(broadcast_shape),
      axes: [-1]
    )
    |> Nx.argmin(axis: 1)
  end

  defnp initialize_matrices(data, opts \\ []) do
    {n, _} = Nx.shape(data)
    self_preference = opts[:self_preference]

    {similarity_matrix, affinity_matrix} =
      initialize_similarities(data, self_preference: self_preference)

    zero = Nx.tensor(0, type: Nx.type(similarity_matrix))
    availability_matrix = Nx.broadcast(zero, {n, n})
    responsibility_matrix = Nx.broadcast(zero, {n, n})

    {availability_matrix, responsibility_matrix, similarity_matrix, affinity_matrix}
  end

  defn initialize_similarities(data, opts \\ []) do
    {n, dims} = Nx.shape(data)
    self_preference = opts[:self_preference]
    t1 = Nx.reshape(data, {1, n, dims}) |> Nx.broadcast({n, n, dims})
    t2 = Nx.reshape(data, {n, 1, dims}) |> Nx.broadcast({n, n, dims})

    dist =
      (-1 * Scholar.Metrics.Distance.squared_euclidean(t1, t2, axes: [-1]))
      |> Nx.as_type(to_float_type(data))

    fill_in =
      cond do
        self_preference == false ->
          Nx.broadcast(Nx.median(dist), {n})

        true ->
          if Nx.size(self_preference) == 1,
            do: Nx.broadcast(self_preference, {n}),
            else: self_preference
      end

    s_modified = dist |> Nx.put_diagonal(fill_in)
    {s_modified, dist}
  end
end
