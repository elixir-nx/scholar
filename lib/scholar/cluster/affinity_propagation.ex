defmodule Scholar.Cluster.AffinityPropagation do
  @moduledoc """
  Model representing affinity propagation clustering.
  """

  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container,
           containers: [:labels, :cluster_centers_indices, :affinity_matrix, :cluster_centers]}
  defstruct [:labels, :cluster_centers_indices, :affinity_matrix, :cluster_centers]

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

  To compute clusters and labels you need to use compute_values, so the usual usecase is

  Scholar.Cluster.AffinityPropagation.fit(data, opts) |> Scholar.Cluster.AffinityPropagation.compute_values()

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns:

    * `:s` - Similarity matrix, it is an Affinity matrix with diagonal filled with `:self_preference`.

    * `:affinity_matrix` - Affinity matrix. It is a negated squared euclidean distance of each pair of points.

    * `:diagonals` - Indicates which data points can be treated as exemplares.

    * `:data` - Provided data.

  ## Examples

      iex> seed = 42
      iex> x = Nx.tensor([[12,5,78,2], [1,5,7,32], [1,3,6,1], [1,2,5,2]])
      iex> Scholar.Cluster.AffinityPropagation.fit(x, seed: seed)
      {Nx.tensor(
        [
          [-939.5, -6062.0, -5310.0, -5459.0],
          [-6062.0, -939.5, -966.0, -913.0],
          [-5310.0, -966.0, -939.5, -3.0],
          [-5459.0, -913.0, -3.0, -939.5]
        ]
      ),
      Nx.tensor(
        [
          [0.0, -6062.0, -5310.0, -5459.0],
          [-6062.0, 0.0, -966.0, -913.0],
          [-5310.0, -966.0, 0.0, -3.0],
          [-5459.0, -913.0, -3.0, 0.0]
        ]
      ),
      Nx.tensor(
        [1, 0, 0, 1], type: :u8
      ),
      Nx.tensor(
        [
          [12, 5, 78, 2],
          [1, 5, 7, 32],
          [1, 3, 6, 1],
          [1, 2, 5, 2]
        ]
      )}
  """
  deftransform fit(data, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    opts = Keyword.update(opts, :self_preference, false, fn x -> x end)
    seed = Keyword.get_lazy(opts, :seed, &System.system_time/0)
    fit_n(data, seed, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(data, seed, opts) do
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
          (Nx.Constants.smallest_positive_normal(Nx.type(s)) * 100 + 2.220446049250313e-16 * s)

    range = Nx.iota({n})

    {_final_a, _final_r, diagonals, _, _} =
      while {a = initial_a, r = initial_r,
             _diagonals = Nx.broadcast(Nx.tensor(0, type: {:u, 8}), {n}), s = s, i = 0},
            i < iterations do
        temp = a + s

        indices = Nx.argmax(temp, axis: 1)

        y = Nx.reduce_max(temp, axes: [1])

        neg_inf = Nx.Constants.neg_infinity(Nx.Type.to_floating(Nx.type(a)))
        inf = Nx.Constants.infinity(Nx.Type.to_floating(Nx.type(a)))

        neg_infinities = Nx.broadcast(neg_inf, {n})

        max_indices =
          Nx.stack([range, indices])
          |> Nx.transpose()

        temp = Nx.indexed_put(temp, max_indices, neg_infinities)

        y2 = Nx.reduce_max(temp, axes: [1])

        temp = s - Nx.new_axis(y, -1)

        temp =
          Nx.indexed_put(
            temp,
            max_indices,
            Nx.flatten(Nx.gather(s, max_indices) - y2)
          )

        temp = temp * (1 - damping_factor)

        r = r * damping_factor + temp

        temp = Nx.max(r, 0)

        temp = Nx.put_diagonal(temp, Nx.take_diagonal(r))

        temp = temp - Nx.sum(temp, axes: [0])

        a_change = Nx.take_diagonal(temp)

        temp = Nx.clip(temp, 0, inf)

        temp = Nx.put_diagonal(temp, a_change)

        temp = temp * (1 - damping_factor)

        a = a * damping_factor - temp

        diagonals = Nx.take_diagonal(a) + Nx.take_diagonal(r) > 0

        {a, r, diagonals, s, i + 1}
      end

    {s, affinity_matrix, diagonals, data}
  end

  @doc """
  Computes labels and centers of clusterization.

  ## Return Values

    The function returns a struct with the following parameters:

    * `:affinity_matrix` - Affinity matrix. It is a negated squared euclidean distance of each pair of points.

    * `:clusters_centers` - Cluster centers from the initial data.

    * `:cluster_centers_indices` - Indices of cluster centers.

    * `:labels` - Labels of each point.

  ## Examples

      iex> seed = 42
      iex> x = Nx.tensor([[12,5,78,2], [1,5,7,32], [1,3,6,1], [1,2,5,2]])
      iex> {s, affinity_matrix, diagonals, data} = Scholar.Cluster.AffinityPropagation.fit(x, seed: seed)
      iex> Scholar.Cluster.AffinityPropagation.compute_values({s, affinity_matrix, diagonals, data})
      %Scholar.Cluster.AffinityPropagation{
        labels: Nx.tensor([0, 1, 1, 1]),
        cluster_centers_indices: Nx.tensor([0, 3]),
        affinity_matrix: Nx.tensor(
          [
            [0.0, -6062.0, -5310.0, -5459.0],
            [-6062.0, 0.0, -966.0, -913.0],
            [-5310.0, -966.0, 0.0, -3.0],
            [-5459.0, -913.0, -3.0, 0.0]
          ]),
        cluster_centers: Nx.tensor(
          [
            [12, 5, 78, 2],
            [1, 2, 5, 2]
          ]
        )
      }
  """
  def compute_values({s, affinity_matrix, diagonals, data}) do
    k = Nx.sum(diagonals, axes: [0]) |> Nx.to_number()

    {cluster_centers, cluster_centers_indices, labels} =
      if k > 0 do
        indices =
          Nx.multiply(Nx.flatten(diagonals), Nx.add(Nx.iota(Nx.shape(diagonals)), 1))
          |> Nx.to_flat_list()
          |> Enum.filter(fn x -> x != 0 end)
          |> Nx.tensor()
          |> Nx.subtract(1)
          |> Nx.as_type({:s, 64})

        c = Nx.argmax(Nx.take(s, indices, axis: 1), axis: 1)
        c = Nx.indexed_put(c, Nx.new_axis(indices, -1), Nx.iota({k}))
        {Nx.take(data, indices), indices, c}
      else
        {n, _} = Nx.shape(s)

        {Nx.tensor([-1], type: Nx.type(s)), Nx.broadcast(Nx.tensor(-1, type: Nx.type(s)), {n}),
         Nx.tensor([-1], type: Nx.type(s))}
      end

    %__MODULE__{
      affinity_matrix: affinity_matrix,
      cluster_centers_indices: cluster_centers_indices,
      cluster_centers: cluster_centers,
      labels: labels
    }
  end

  @doc """
  Predict the closest cluster each sample in `x` belongs to.

  ## Examples

      iex> seed = 42
      iex> x = Nx.tensor([[12,5,78,2], [1,5,7,32], [1,3,6,1], [1,2,5,2]])
      iex> {s, affinity_matrix, diagonals, data} = Scholar.Cluster.AffinityPropagation.fit(x, seed: seed)
      iex> model = Scholar.Cluster.AffinityPropagation.compute_values({s, affinity_matrix, diagonals, data})
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
    %{type: type} = data
    {n, _} = Nx.shape(data)
    self_preference = opts[:self_preference]
    zero = Nx.tensor(0, type: Nx.Type.to_floating(type))
    availability_matrix = Nx.broadcast(zero, {n, n})
    responsibility_matrix = Nx.broadcast(zero, {n, n})

    {similarity_matrix, affinity_matrix} =
      initialize_similarities(data, self_preference: self_preference)

    {availability_matrix, responsibility_matrix, similarity_matrix, affinity_matrix}
  end

  defn initialize_similarities(data, opts \\ []) do
    {n, dims} = Nx.shape(data)
    self_preference = opts[:self_preference]
    t1 = Nx.reshape(data, {1, n, dims})
    t2 = Nx.reshape(data, {n, 1, dims})

    dist =
      (t1 - t2)
      |> Nx.pow(2)
      |> Nx.sum(axes: [2])
      |> Nx.multiply(-1)
      |> Nx.as_type(to_float_type(data))

    fill_in =
      case self_preference do
        false ->
          Nx.broadcast(Nx.median(dist), {n})

        _ ->
          if Nx.size(self_preference) == 1,
            do: Nx.broadcast(self_preference, {n}),
            else: self_preference
      end

    s_modified = dist |> Nx.put_diagonal(fill_in)
    {s_modified, dist}
  end
end
