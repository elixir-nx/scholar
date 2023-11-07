defmodule Scholar.Metrics.Clustering do
  @moduledoc """
  Metrics related to clustering algorithms.
  """

  import Nx.Defn
  import Scholar.Shared

  opts = [
    num_clusters: [
      required: true,
      type: :pos_integer,
      doc: "Number of clusters in clustering."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Compute the Silhouette Coefficient for each sample.

  The silhouette value is a measure of how similar an object is to its own cluster (cohesion)
  compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high
  value indicates that the object is well matched to its own cluster and poorly
  matched to neighboring clusters. If most objects have a high value, then the
  clustering configuration is appropriate. If many points have a low or negative
  value, then the clustering configuration may have too many or too few clusters.

  Time complexity of silhouette score is $O(N^2)$ where $N$ is the number of samples.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

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
        [0.0, -0.9782054424285889, 0.0, -0.18546827137470245, -0.5929659008979797]
      >
  """
  deftransform silhouette_samples(x, labels, opts \\ []) do
    silhouette_samples_n(x, labels, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp silhouette_samples_n(x, labels, opts) do
    verify_num_clusters(x, opts)
    {inner, alone?, outer} = inner_and_outer_dist(x, labels, opts)
    result = (outer - inner) / Nx.max(outer, inner)
    Nx.select(alone?, 0, result)
  end

  @doc """
  Compute the mean Silhouette Coefficient of all samples.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

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
        -0.35132792592048645
      >
  """
  deftransform silhouette_score(x, labels, opts \\ []) do
    silhouette_score_n(x, labels, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp silhouette_score_n(x, labels, opts) do
    Nx.mean(silhouette_samples(x, labels, opts))
  end

  defnp inner_and_outer_dist(x, labels, opts) do
    num_clusters = opts[:num_clusters]
    num_samples = Nx.axis_size(x, 0)
    inf = Nx.Constants.infinity(to_float_type(x))
    pairwise_dist = Scholar.Metrics.Distance.pairwise_euclidean(x)
    membership_mask = Nx.reshape(labels, {num_samples, 1}) == Nx.iota({1, num_clusters})
    cluster_size = membership_mask |> Nx.sum(axes: [0]) |> Nx.reshape({1, num_clusters})
    dist_in_cluster = Nx.dot(pairwise_dist, membership_mask)
    mean_dist_in_cluster = dist_in_cluster / cluster_size

    alone? = (cluster_size == 1) |> Nx.squeeze() |> Nx.take(labels)

    inner_dist =
      (dist_in_cluster / Nx.max(cluster_size - 1, 1))
      |> Nx.take_along_axis(Nx.reshape(labels, {num_samples, 1}), axis: 1)
      |> Nx.squeeze(axes: [1])

    outer_dist =
      membership_mask
      |> Nx.select(inf, mean_dist_in_cluster)
      |> Nx.reduce_min(axes: [1])

    {inner_dist, alone?, outer_dist}
  end

  deftransformp verify_num_clusters(x, opts) do
    {num_samples, _} = Nx.shape(x)

    unless opts[:num_clusters] <= num_samples do
      raise ArgumentError,
            "expected :num_clusters to to be a positive integer in range 1 to #{inspect(num_samples)}, got: #{inspect(opts[:num_clusters])}"
    end
  end
end
