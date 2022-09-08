defmodule Scholar.Metrics.Clustering do
  @moduledoc """
  Metrics related to clustering algorithms.
  """

  import Nx.Defn

  defnp inner_dist(x, labels) do
    {num_samples, num_features} = Nx.shape(x)

    expanded_points =
      Nx.tile(x, [1, num_samples]) |> Nx.reshape({num_samples, num_samples, num_features})

    labels_vectorize = Nx.broadcast(labels, {num_samples, num_samples}) |> Nx.transpose()

    mask_zero_one =
      labels_vectorize ==
        Nx.tile(labels, [1, num_samples]) |> Nx.reshape({num_samples, num_samples})

    denominator = Nx.sum(mask_zero_one, axes: [1]) |> Nx.reshape({num_samples, 1}) |> Nx.squeeze()
    indices_to_zero = Nx.equal(denominator, 1)
    denominator = Nx.select(Nx.greater(denominator, 1), denominator - 1, 1)

    mask =
      mask_zero_one
      |> Nx.select(
        Nx.tile(Nx.iota({num_samples}), [num_samples, 1]),
        Nx.broadcast(Nx.iota({num_samples, 1}), {num_samples, num_samples})
      )

    points = Nx.take(x, mask)

    {Nx.sum(
       Scholar.Metrics.Distance.euclidean(
         expanded_points,
         points,
         axes: [2]
       ),
       axes: [1]
     ) / denominator, indices_to_zero}
  end

  defnp outer_dist(x, labels, opts \\ []) do
    num_clusters = opts[:num_clusters]
    {num_samples, num_features} = Nx.shape(x)
    inf = Nx.Constants.infinity()

    x_a =
      x
      |> Nx.reshape({1, num_samples, num_features})
      |> Nx.broadcast({num_samples, num_samples, num_features})

    x_b =
      x
      |> Nx.reshape({num_samples, 1, num_features})
      |> Nx.broadcast({num_samples, num_samples, num_features})

    pairwise_dist = Scholar.Metrics.Distance.euclidean(x_a, x_b, axes: [2])
    labels_vectorize = Nx.tile(labels, [num_clusters, 1])
    Nx.iota({num_clusters, num_samples}, axis: 0)
    mask = labels_vectorize == Nx.iota({num_clusters, num_samples}, axis: 0)
    denominator = mask |> Nx.sum(axes: [1]) |> Nx.broadcast({num_samples, num_clusters})
    results = Nx.dot(mask, pairwise_dist) |> Nx.transpose()
    results = Nx.select(Nx.transpose(mask), inf, results)
    results = results / denominator
    Nx.reduce_min(results, axes: [1])
  end

  deftransformp verify_num_clusters(x, opts) do
    {num_samples, _} = Nx.shape(x)

    unless opts[:num_clusters] do
      raise ArgumentError,
            "missing option :num_clusters"
    end

    unless is_integer(opts[:num_clusters]) and opts[:num_clusters] > 0 and
             opts[:num_clusters] <= num_samples do
      raise ArgumentError,
            "expected :num_clusters to to be a positive integer in range 1 to #{inspect(num_samples)}, got: #{inspect(opts[:num_clusters])}"
    end
  end

  @doc """
  Compute the Silhouette Coefficient for each sample.

  The silhouette value is a measure of how similar an object is to its own cluster (cohesion)
  compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high
  value indicates that the object is well matched to its own cluster and poorly
  matched to neighboring clusters. If most objects have a high value, then the
  clustering configuration is appropriate. If many points have a low or negative
  value, then the clustering configuration may have too many or too few clusters.

  ## Options

  * `:num_clusters` - Number of clusters in clustering. Required.

  ## Examples

      iex> x = Nx.tensor([[0, 0], [1, 0], [1, 1], [3, 3], [4, 4.5]])
      iex> labels = Nx.tensor([0, 0, 0, 1, 1])
      iex> Scholar.Metrics.Clustering.silhouette_samples(x, labels, num_clusters: 2)
      #Nx.Tensor<
        f32[5]
        [0.7647753357887268, 0.7781199216842651, 0.6754303574562073, 0.49344196915626526, 0.6627992987632751]
      >

      iex> x = Nx.tensor([[0.1, 0], [0, 1], [22, 65], [42, 3], [4.2, 51]])
      iex> labels = Nx.tensor([0, 1, 2, 1, 1])
      iex> Scholar.Metrics.Clustering.silhouette_samples(x, labels, num_clusters: 3)
      #Nx.Tensor<
        f32[5]
        [0.0, -0.9782054424285889, 0.0, -0.18546819686889648, -0.5929657816886902]
      >
  """

  defn silhouette_samples(x, labels, opts \\ []) do
    verify_num_clusters(x, opts)
    outer = outer_dist(x, labels, opts)
    {inner, indices_to_zero} = inner_dist(x, labels)
    result = (outer - inner) / Nx.max(outer, inner)
    Nx.select(indices_to_zero, 0, result)
  end

  @doc """
  Compute the mean Silhouette Coefficient of all samples.

    ## Options

  * `:num_clusters` - Number of clusters in clustering. Required.

  ## Examples

      iex> x = Nx.tensor([[0, 0], [1, 0], [1, 1], [3, 3], [4, 4.5]])
      iex> labels = Nx.tensor([0, 0, 0, 1, 1])
      iex> Scholar.Metrics.Clustering.silhouette_score(x, labels, num_clusters: 2)
      #Nx.Tensor<
        f32
        0.6749133467674255
      >

      iex> x = Nx.tensor([[0.1, 0], [0, 1], [22, 65], [42, 3], [4.2, 51]])
      iex> labels = Nx.tensor([0, 1, 2, 1, 1])
      iex> Scholar.Metrics.Clustering.silhouette_score(x, labels, num_clusters: 3)
      #Nx.Tensor<
        f32
        -0.35132789611816406
      >

  """

  defn silhouette_score(x, labels, opts \\ []) do
    Nx.mean(silhouette_samples(x, labels, opts))
  end
end
