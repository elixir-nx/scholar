defmodule Scholar.Cluster.KMeans do
  @moduledoc """
  K-Means algorithm
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :bias]}
  defstruct [:coefficients, :bias, :mode]

  @doc """
  Fits a K-Means model for sample inputs `x`


  ## Options

    * `:xxx` - yyy

    * `:xxx` - yyy

    * `:xxx` - yyy

  """

  defn fit(x, opts \\ []) do
    opts =
      keyword!(
        opts,
        [num_clusters: 8,
        iterations: 300,
        num_runs: 10,
        tol: 1.0e-4,
        random_state: -1,
        init: :k_means_plus_plus]
      )

    # verify(x, opts)
    if opts[:random_state] == -1, do: nil, else: :rand.seed(:exsss, opts[:random_state])
    x = x |> Nx.as_type({:f, 32})
    inf = Nx.Constants.infinity({:f, 32})
    {num_samples, num_features} = Nx.shape(x)
    num_clusters = opts[:num_clusters]


    initial_centroids = initialize_centroids(x, opts)
    distance = inf

    {i, _, _,_, final_inertia, final_centroids, nearest_centroids} =
      while {i = 0, x, previous_iteration_centroids = Nx.broadcast(inf, initial_centroids), distance,
             inertia = 0.0, initial_centroids,
             nearest_centroids = Nx.broadcast(-1, {num_samples})},
            Nx.less(i, opts[:iterations]) and
              Nx.greater(
                distance,
                opts[:tol]
              ) do
        previous_iteration_centroids = initial_centroids

        {inertia_for_centroids, min_inertia} = calculate_inetria(x, initial_centroids, num_clusters)
        nearest_centroids = Nx.argmin(inertia_for_centroids, axis: 0)

        inertia = Nx.sum(min_inertia)

        group_masks =
          Nx.equal(
            Nx.iota({num_clusters, 1}),
            Nx.reshape(nearest_centroids, {1, num_samples})
          )

        group_sizes = Nx.sum(group_masks, axes: [1], keep_axes: true)

        initial_centroids =
          Nx.multiply(
            Nx.reshape(group_masks, {num_clusters, num_samples, 1}),
            Nx.reshape(x, {1, num_samples, num_features})
          )
          |> Nx.sum(axes: [1])
          |> Nx.divide(group_sizes)

        distance = Nx.sum(Nx.abs(initial_centroids - previous_iteration_centroids))
        {i + 1, x, previous_iteration_centroids, distance, inertia, initial_centroids, nearest_centroids}
      end

    {final_inertia, final_centroids, i, nearest_centroids}
  end

  defnp initialize_centroids(x, opts) do
    num_clusters = opts[:num_clusters]
    {num_samples, _num_features} = Nx.shape(x)
    num_runs = opts[:num_runs]

    case opts[:init] do
      :random ->
        # 0..(num_samples - 1)
        # |> Enum.take_random(num_clusters)
        # |> Nx.to_tensor()
        # Nx.random_uniform({num_clusters}, 0, num_samples, type: {:s, 64})
        # |> then(&Nx.take_along_axis(x, &1, axis: 0))
          Nx.iota({num_runs, num_samples}, axis: 1)
          |> Nx.shuffle(axis: 1)
          |> Nx.slice_along_axis(0, num_runs, axis: 1)
          |> then(&Nx.take(x, &1))

      :k_means_plus_plus ->
        k_means_plus_plus(x, num_clusters)
    end
  end

  defn generate_idx(mod_weights) do
    rand = Nx.random_uniform({1}, 0, 1, type: {:f, 32})
    val = rand * Nx.sum(mod_weights)
    cumulative_weights = Nx.cumulative_sum(mod_weights)
    Nx.greater_equal(cumulative_weights, val) |> Nx.argmax(tie_break: :low)
  end

  defnp calculate_inetria(x, centroids, num_clusters) do
    {num_samples, num_features} = Nx.shape(x)

    modified_centroids =
      centroids
      |> Nx.reshape({num_clusters, 1, num_features})
      |> Nx.broadcast({num_clusters, num_samples, num_features})
      |> Nx.reshape({num_clusters * num_samples, num_features})

    inertia_for_centroids =
      Scholar.Cluster.Utils.squared_euclidean(
        Nx.tile(x, [num_clusters, 1]),
        modified_centroids,
        axes: 1
      )
      |> Nx.reshape({num_clusters, num_samples})

    {inertia_for_centroids, Nx.reduce_min(inertia_for_centroids, axes: [0])}
  end

  defnp find_new_centroid(weights, x, centroid_mask) do
    mod_weights = weights * centroid_mask
    idx = generate_idx(mod_weights)
    {x[idx], idx}
  end

  defnp k_means_plus_plus(x, num_clusters) do
    idx = 0
    # inf = 1_000_000_000.0
    inf = Nx.Constants.infinity()
    {num_samples, num_features} = Nx.shape(x)
    centroids = Nx.broadcast(inf, {num_clusters, num_features})
    inertia = Nx.broadcast(0.0, {num_samples})
    centroid_mask = Nx.broadcast(1, {num_samples})

    first_centroid_idx = Nx.random_uniform({1}, 0, num_samples - 1, type: {:u, 32})

    first_centroid = x[first_centroid_idx] |> Nx.new_axis(0)
    centroids = Nx.put_slice(centroids, [idx, 0], first_centroid)
    idx = idx + 1
    centroid_mask = Nx.put_slice(centroid_mask, [first_centroid_idx[0]], Nx.tensor([0]))

    {_, _, _, _, final_centroids} =
      while {idx, centroid_mask, x, inertia, centroids},
            Nx.less(idx, num_clusters) do
        {_inertia_for_centroids, min_inertia} = calculate_inetria(x, centroids, num_clusters)
        {new_centroid, centroid_idx} = find_new_centroid(min_inertia, x, num_clusters)
        centroids = Nx.put_slice(centroids, [idx, 0], Nx.new_axis(new_centroid, 0))
        centroid_mask = Nx.put_slice(centroid_mask, [centroid_idx], Nx.tensor([0]))
        {idx + 1, centroid_mask, x, inertia, centroids}
      end

    final_centroids
  end

  # Function checks validity of the provided data

  deftransformp verify(x, opts) do


    unless opts[:num_classes] do
      raise ArgumentError, "missing option :num_classes"
    end

    unless is_integer(opts[:num_classes]) and opts[:num_classes] > 0 do
      raise ArgumentError,
            "expected :num_classes to be a positive integer, got: #{inspect(opts[:num_classes])}"
    end

    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    unless is_number(opts[:learning_rate]) and opts[:learning_rate] > 0 do
      raise ArgumentError,
            "expected :learning_rate to be a positive number, got: #{inspect(opts[:learning_rate])}"
    end

    unless is_integer(opts[:iterations]) and opts[:iterations] > 0 do
      raise ArgumentError,
            "expected :iterations to be a positive integer, got: #{inspect(opts[:iterations])}"
    end
  end

  #
  @doc """
  Makes predictions with the given model on inputs `x`.
  """

  # defn predict(%__MODULE__{mode: mode} = _model, x) do
  #   {x, mode}
  # end
end

#
IO.inspect(
  Scholar.Cluster.KMeans.fit(
    Nx.tensor([[1, 5], [2, 3], [2, 5], [1, 3], [6, 2], [7, 1.5], [6.5, 4]]), num_clusters: 2, init: :random
  )
)
