defmodule Scholar.Cluster.BIRCH do
  @moduledoc """
  TODO
  """
  import Nx.Defn
  import Scholar.Shared
  require Nx

  @derive {Nx.Container, containers: [:cf_nodes]}
  defstruct [:cf_nodes]

  opts = [
    num_clusters: [
      type: :pos_integer,
      default: 3,
      doc: "The number of clusters to form as well as the number of centroids to generate."
    ],
    threshold: [
      type: :float,
      default: 0.5,
      doc: """
      The radius of the subcluster obtained by merging a new sample and the closest 
      subcluster should be lesser than the threshold. Otherwise a new subcluster is started. 
      Setting this value to be very low promotes splitting and vice-versa.
      """
    ],
    branching_factor: [
      type: :pos_integer,
      default: 50,
      doc: """
      Maximum number of CF subclusters in each node. If a new samples enters such that the number 
      of subclusters exceed the branching_factor then that node is split into two nodes with the 
      subclusters redistributed in each. The parent subcluster of that node is removed and two new 
      subclusters are added as parents of the 2 split nodes.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a BIRCH model for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:labels` - Labels of each point.

  ## Examples
  """
  deftransform fit(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected input tensor to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    num_samples = Nx.axis_size(x, 0)
    opts = NimbleOptions.validate!(opts, @opts_schema)

    unless opts[:num_clusters] <= num_samples do
      raise ArgumentError,
            "invalid value for :num_clusters option: expected positive integer between 1 and #{inspect(num_samples)}, got: #{inspect(opts[:num_clusters])}"
    end
    fit_n(x, opts)
  end

  defnp fit_n(%Nx.Tensor{shape: {num_samples, num_features}} = x, opts) do
    x = to_float(x)
    num_clusters = opts[:num_clusters]
    threshold = opts[:threshold]

    # %__MODULE__{
    #   labels: [0, 1, 0, ...],
    # }
  end

  defnp initialize_centroids(x, key, opts) do
    num_clusters = opts[:num_clusters]
    {num_samples, _num_features} = Nx.shape(x)
    num_runs = opts[:num_runs]

    case opts[:init] do
      :random ->
        nums = Nx.iota({num_runs, num_samples}, axis: 1)
        {temp, _} = Nx.Random.shuffle(key, nums, axis: 1)

        temp
        |> Nx.slice_along_axis(0, num_clusters, axis: 1)
        |> then(&Nx.take(x, &1))

      :k_means_plus_plus ->
        k_means_plus_plus(x, num_clusters, num_runs, key)
    end
  end

  defnp split_node(node, threshold, branching_factor) do
    
  end

  defnp compute_labels(x, centroids, num_clusters) do
    # TO DO
  end

  @doc """
  Makes predictions with the given `model` on inputs `x`.

  ## Return Values

    It returns a tensor with clusters corresponding to the input.

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> model = Scholar.Cluster.KMeans.fit(x, num_clusters: 2, key: key)
      iex> Scholar.Cluster.KMeans.predict(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      Nx.tensor(
        [1, 0]
      )
  """
  defn predict(%__MODULE__{clusters: clusters} = _model, x) do
    assert_same_shape!(x[0], clusters[0])

    Scholar.Metrics.Distance.pairwise_squared_euclidean(clusters, x) |> Nx.argmin(axis: 0)
  end


  defn transform(%__MODULE__{clusters: clusters} = _model, x) do
    Scholar.Metrics.Distance.pairwise_euclidean(x, clusters)
  end
end


