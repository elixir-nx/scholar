defmodule Scholar.Cluster.SpectralClustering do
  @moduledoc """
  Spectral clustering.

  Applies clustering to a projection of the normalized graph Laplacian [1].
  Instead of clustering the points in their original space, the algorithm
  builds an affinity (similarity) matrix between all pairs of samples,
  computes the eigenvectors associated with the smallest eigenvalues of its
  normalized graph Laplacian, and runs k-means on that spectral embedding.

  In practice spectral clustering is very useful when the individual
  clusters are non-convex (e.g. nested circles), a setting where plain
  k-means on the original space performs poorly [2].

  The eigendecomposition of the dense `{num_samples, num_samples}` Laplacian
  gives a time complexity of $O(N^3)$ and a memory complexity of $O(N^2)$,
  where $N$ is the number of samples.

  References:

  * [1] - [A Tutorial on Spectral Clustering, Ulrike von Luxburg](https://arxiv.org/abs/0711.0189)
  * [2] - [On Spectral Clustering: Analysis and an algorithm, Ng, Jordan & Weiss](https://proceedings.neurips.cc/paper_files/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf)
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:labels, :embedding]}
  defstruct [:labels, :embedding]

  opts = [
    num_clusters: [
      required: true,
      type: :pos_integer,
      doc: "The number of clusters to form, as well as the dimension of the spectral embedding."
    ],
    affinity: [
      type: {:in, [:rbf, :precomputed]},
      default: :rbf,
      doc: """
      How to construct the affinity matrix, either of:

      * `:rbf` - construct the affinity matrix using a radial basis function (RBF) kernel,
        $\\exp(-\\gamma \\|x_i - x_j\\|^2)$.

      * `:precomputed` - interpret the input as a precomputed affinity matrix, i.e.
        a symmetric `{num_samples, num_samples}` tensor of non-negative similarities.
      """
    ],
    gamma: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0,
      doc: "Kernel coefficient for the `:rbf` affinity. Ignored for `:precomputed`."
    ],
    num_runs: [
      type: :pos_integer,
      default: 10,
      doc: """
      Number of times the k-means algorithm will be run on the spectral embedding
      with different centroid seeds. The final result is the best output in terms
      of inertia.
      """
    ],
    max_iterations: [
      type: :pos_integer,
      default: 300,
      doc: "Maximum number of iterations of the k-means algorithm for a single run."
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Determines random number generation for centroid initialization of the
      k-means step. If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Performs spectral clustering on the sample inputs `x`.

  If `affinity: :rbf` (the default), `x` is expected to be a
  `{num_samples, num_features}` tensor of samples. If `affinity: :precomputed`,
  `x` is expected to be a symmetric `{num_samples, num_samples}` affinity matrix.

  Self-loops are discarded when building the graph Laplacian, i.e. the diagonal
  of the affinity matrix is ignored, matching `scipy.sparse.csgraph.laplacian`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:labels` - Cluster labels of each point.

    * `:embedding` - The `{num_samples, num_clusters}` spectral embedding, i.e. the
      eigenvectors of the normalized graph Laplacian associated with its
      `num_clusters` smallest eigenvalues, rescaled as in the relaxed Ncut problem
      ($u = D^{-1/2} v$) and with a deterministic sign (the entry with the largest
      absolute value of each eigenvector is positive).

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> x = Nx.tensor([[0.0, 0.0], [0.1, 0.1], [3.0, 3.0], [3.1, 3.1]])
      iex> Scholar.Cluster.SpectralClustering.fit(x, num_clusters: 2, key: key).labels
      #Nx.Tensor<
        s32[4]
        [0, 0, 1, 1]
      >
  """
  deftransform fit(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected input tensor to have shape {num_samples, num_features} or " <>
              "{num_samples, num_samples}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)
    num_samples = Nx.axis_size(x, 0)

    if num_samples < 2 do
      raise ArgumentError, "expected at least 2 samples, got: #{num_samples}"
    end

    if opts[:affinity] == :precomputed and Nx.axis_size(x, 1) != num_samples do
      raise ArgumentError,
            "expected a square affinity matrix for affinity: :precomputed, " <>
              "got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    unless opts[:num_clusters] <= num_samples do
      raise ArgumentError,
            "invalid value for :num_clusters option: expected positive integer " <>
              "between 1 and #{inspect(num_samples)}, got: #{inspect(opts[:num_clusters])}"
    end

    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)

    # The stopping criterion of Nx.LinAlg.eigh must match the input precision:
    # a criterion below the floating-point resolution makes the iteration
    # accumulate noise past convergence.
    eigh_eps = if Nx.type(x) == {:f, 64}, do: 1.0e-11, else: 1.0e-8

    {labels, embedding} =
      fit_n(x, key,
        num_clusters: opts[:num_clusters],
        affinity: opts[:affinity],
        gamma: opts[:gamma],
        eigh_eps: eigh_eps,
        num_runs: opts[:num_runs],
        max_iterations: opts[:max_iterations]
      )

    %__MODULE__{labels: labels, embedding: embedding}
  end

  defnp fit_n(x, key, opts) do
    embedding =
      spectral_embedding(x,
        num_clusters: opts[:num_clusters],
        affinity: opts[:affinity],
        gamma: opts[:gamma],
        eigh_eps: opts[:eigh_eps]
      )

    model =
      Scholar.Cluster.KMeans.fit(embedding,
        num_clusters: opts[:num_clusters],
        num_runs: opts[:num_runs],
        max_iterations: opts[:max_iterations],
        key: key
      )

    {model.labels, embedding}
  end

  defnp spectral_embedding(x, opts) do
    affinity =
      case opts[:affinity] do
        :rbf ->
          Nx.exp(-opts[:gamma] * Scholar.Metrics.Distance.pairwise_squared_euclidean(x))

        :precomputed ->
          Scholar.Shared.to_float(x)
      end

    num_samples = Nx.axis_size(affinity, 0)
    num_clusters = opts[:num_clusters]

    # Self-loops are discarded before computing the degrees, matching
    # scikit-learn (`scipy.sparse.csgraph.laplacian`).
    eye = Nx.eye(num_samples, type: Nx.type(affinity))
    affinity = affinity * (1 - eye)

    degrees = Nx.sum(affinity, axes: [1])
    dd = Nx.select(degrees == 0, 1, Nx.sqrt(degrees))

    # Symmetric normalized Laplacian L = I - D^-1/2 A D^-1/2. Isolated vertices
    # keep a unit diagonal, as scikit-learn does via `_set_diag(laplacian, 1)`.
    laplacian = -(affinity / dd / Nx.new_axis(dd, -1)) + eye

    # The division can leave tiny floating-point asymmetries; eigh requires an
    # exactly symmetric input.
    laplacian = (laplacian + Nx.transpose(laplacian)) / 2

    # eigh returns unit eigenvectors, but not necessarily sorted, so the
    # num_clusters smallest eigenvalues are selected explicitly.
    {eigenvalues, eigenvectors} = Nx.LinAlg.eigh(laplacian, eps: opts[:eigh_eps])
    order = Nx.argsort(eigenvalues, direction: :asc)
    vectors = Nx.take(eigenvectors, order[0..(num_clusters - 1)//1], axis: 1)

    # Recover u = D^-1/2 v, the solution of the relaxed Ncut problem.
    embedding = vectors / Nx.new_axis(dd, -1)

    # Deterministic sign: the largest-magnitude entry of each vector is made
    # positive, as in scikit-learn's `_deterministic_vector_sign_flip`.
    max_abs = embedding |> Nx.abs() |> Nx.argmax(axis: 0, keep_axis: true)
    signs = embedding |> Nx.take_along_axis(max_abs, axis: 0) |> Nx.sign()
    embedding * signs
  end
end
