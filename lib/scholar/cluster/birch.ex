defmodule Scholar.Cluster.BIRCH do
  @moduledoc """
  BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).

  BIRCH incrementally builds a height-balanced tree of Clustering Features (CFs).
  Each point is inserted into the closest leaf subcluster if the resulting subcluster
  radius stays below `:threshold`; otherwise a new subcluster is created. Nodes that
  exceed `:branching_factor` are split. The leaf subcluster centroids are then reduced
  to the requested number of clusters with agglomerative (hierarchical) clustering.

  Because the tree is built by inserting one sample at a time, the CF-tree is
  constructed on the host in Elixir while the numeric work (distances, radii, the
  global clustering step) is done with `Nx`. Each insertion reads a scalar back from
  the tensor backend, so the tree-building phase does not benefit from a compiled
  backend the way the rest of Scholar does.

  Unlike scikit-learn's `Birch`, which reduces the leaf subclusters with agglomerative
  clustering, the global step here uses `Scholar.Cluster.KMeans`. Labels may therefore
  differ from scikit-learn on data where the two disagree, and the step is seeded by
  the `:key` option.

  Building the CF-tree takes a single pass over the data: inserting one sample costs
  $O(BDH)$, where $B$ is `:branching_factor`, $D$ is the number of features and $H$ is
  the height of the tree, for $O(NBDH)$ overall with $N$ samples. Space complexity is
  $O(MD)$ where $M$ is the number of leaf subclusters, which is typically far smaller
  than $N$ — this is what makes BIRCH suitable for large datasets. Labelling the input
  at the end is $O(NMD)$.

  Reference:

  * [BIRCH: An Efficient Data Clustering Method for Very Large Databases](https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf)
  """

  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:subcluster_centers, :subcluster_labels, :labels]}
  defstruct [:subcluster_centers, :subcluster_labels, :labels]

  opts = [
    threshold: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 0.5,
      doc: """
      The radius of the subcluster obtained by merging a new sample and the closest subcluster
      must be less than this threshold, otherwise a new subcluster is started.
      """
    ],
    branching_factor: [
      type: :pos_integer,
      default: 50,
      doc: """
      The maximum number of subclusters in each node. If a node exceeds this value,
      it is split into two nodes.
      """
    ],
    num_clusters: [
      type: :pos_integer,
      default: 3,
      doc: "The number of clusters to form from the leaf subclusters."
    ],
    num_runs: [
      type: :pos_integer,
      default: 10,
      doc: """
      Number of times the k-means algorithm will be run on the leaf subcluster centroids
      with different centroid seeds. The final result is the best output in terms of inertia.
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
      Determines random number generation for centroid initialization of the global
      clustering step. If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
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

  * `:subcluster_centers` - Centroids of all the leaf subclusters.

  * `:subcluster_labels` - The cluster each leaf subcluster is assigned to.

  * `:labels` - Cluster label of each point in `x`.

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> model = Scholar.Cluster.BIRCH.fit(x, num_clusters: 2, key: key)
      iex> model.labels
      #Nx.Tensor<
        s32[4]
        [0, 1, 0, 1]
      >
  """
  deftransform fit(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected input tensor to have shape {num_samples, num_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)
    num_samples = Nx.axis_size(x, 0)

    unless opts[:num_clusters] <= num_samples do
      raise ArgumentError,
            "invalid value for :num_clusters option: expected positive integer between 1 and #{num_samples}, got: #{inspect(opts[:num_clusters])}"
    end

    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    x = to_float(x)

    tree =
      Enum.reduce(0..(num_samples - 1), new_leaf(), fn i, root ->
        insert_root(root, x[i], opts[:threshold], opts[:branching_factor])
      end)

    subcluster_centers = tree |> leaf_centroids() |> Nx.stack()
    subcluster_labels = global_labels(subcluster_centers, key, opts)
    labels = assign_labels(subcluster_centers, subcluster_labels, x)

    %__MODULE__{
      subcluster_centers: subcluster_centers,
      subcluster_labels: subcluster_labels,
      labels: labels
    }
  end

  # --- CF-tree construction (host side) ---
  #
  # A node is %{is_leaf: bool, subclusters: [subcluster]}.
  # A subcluster is a Clustering Feature %{n, ls, ss, centroid, sq_norm, child}, where
  #   * n        - number of samples,
  #   * ls       - linear sum of the samples (tensor {num_features}),
  #   * ss       - sum of squared norms of the samples (scalar),
  #   * centroid - LS/n, cached rather than recomputed on every descent,
  #   * sq_norm  - ||centroid||^2, likewise cached,
  #   * child    - child node for non-leaf subclusters, nil for leaf subclusters.

  defp new_leaf, do: %{is_leaf: true, subclusters: []}

  # Insert a sample at the root, growing a new root if the root splits.
  defp insert_root(root, point, threshold, branching_factor) do
    case insert(root, point, threshold, branching_factor) do
      {:ok, node} ->
        node

      {:split, a, b} ->
        %{is_leaf: false, subclusters: [parent_subcluster(a), parent_subcluster(b)]}
    end
  end

  defp insert(%{subclusters: []} = node, point, _threshold, _branching_factor) do
    {:ok, %{node | subclusters: [new_subcluster(point)]}}
  end

  defp insert(node, point, threshold, branching_factor) do
    idx = closest_index(node.subclusters, point)
    closest = Enum.at(node.subclusters, idx)

    if closest.child do
      case insert(closest.child, point, threshold, branching_factor) do
        {:ok, child} ->
          updated = %{add_point(closest, point) | child: child}
          {:ok, %{node | subclusters: List.replace_at(node.subclusters, idx, updated)}}

        {:split, a, b} ->
          # The first half replaces the split subcluster in place and the second half
          # goes to the end of the node. Subcluster order decides which entry wins a
          # distance tie later on, so it has to match scikit-learn's
          # `update_split_subclusters`.
          subclusters =
            node.subclusters
            |> List.replace_at(idx, parent_subcluster(a))
            |> Enum.concat([parent_subcluster(b)])

          maybe_split(%{node | subclusters: subclusters}, branching_factor)
      end
    else
      case merge_point(closest, point, threshold) do
        {:ok, merged} ->
          {:ok, %{node | subclusters: List.replace_at(node.subclusters, idx, merged)}}

        :error ->
          subclusters = node.subclusters ++ [new_subcluster(point)]
          maybe_split(%{node | subclusters: subclusters}, branching_factor)
      end
    end
  end

  defp maybe_split(node, branching_factor) do
    if length(node.subclusters) > branching_factor do
      {a, b} = split_node(node)
      {:split, a, b}
    else
      {:ok, node}
    end
  end

  # Split a node into two by seeding with the farthest pair of subclusters and
  # assigning every subcluster to its nearer seed. Ties go to the second seed, and
  # the first seed is always assigned to itself so that a node whose centroids all
  # coincide (every distance zero) still splits into two non-empty nodes. This
  # matches scikit-learn's `_split_node`.
  defp split_node(node) do
    centroids = node.subclusters |> Enum.map(& &1.centroid) |> Nx.stack()
    dist = Scholar.Metrics.Distance.pairwise_squared_euclidean(centroids)
    {i, j} = farthest_pair(dist)

    closer_to_i =
      dist[i]
      |> Nx.less(dist[j])
      |> Nx.to_flat_list()
      |> List.replace_at(i, 1)

    {group_a, group_b} =
      node.subclusters
      |> Enum.zip(closer_to_i)
      |> Enum.split_with(fn {_sc, near_i} -> near_i == 1 end)

    a = %{is_leaf: node.is_leaf, subclusters: Enum.map(group_a, &elem(&1, 0))}
    b = %{is_leaf: node.is_leaf, subclusters: Enum.map(group_b, &elem(&1, 0))}
    {a, b}
  end

  defp farthest_pair(dist) do
    k = Nx.axis_size(dist, 0)
    flat = Nx.to_number(Nx.argmax(dist))
    {div(flat, k), rem(flat, k)}
  end

  # --- Clustering Feature helpers ---

  defp new_subcluster(point) do
    sq_norm = squared_norm(point)
    %{n: 1, ls: point, ss: sq_norm, centroid: point, sq_norm: sq_norm, child: nil}
  end

  # Absorbing a point into a subcluster along the descent path. The centroid is
  # obtained by dividing, matching scikit-learn's `_CFSubcluster.update`.
  defp add_point(subcluster, point) do
    n = subcluster.n + 1
    ls = Nx.add(subcluster.ls, point)
    centroid = Nx.divide(ls, n)

    %{
      subcluster
      | n: n,
        ls: ls,
        ss: subcluster.ss + squared_norm(point),
        centroid: centroid,
        sq_norm: squared_norm(centroid)
    }
  end

  # Absorbing a point into a leaf subcluster, which only goes ahead if the resulting
  # subcluster still fits inside the threshold. Unlike `add_point/2` the centroid is
  # scaled by the reciprocal and the radius is compared squared — this is
  # scikit-learn's `_CFSubcluster.merge_subcluster`, and the two round differently
  # right at the threshold.
  defp merge_point(subcluster, point, threshold) do
    n = subcluster.n + 1
    ls = Nx.add(subcluster.ls, point)
    ss = subcluster.ss + squared_norm(point)
    centroid = Nx.multiply(Nx.tensor(1 / n, type: Nx.type(ls)), ls)
    sq_norm = squared_norm(centroid)

    if ss / n - sq_norm <= threshold * threshold do
      {:ok, %{subcluster | n: n, ls: ls, ss: ss, centroid: centroid, sq_norm: sq_norm}}
    else
      :error
    end
  end

  # A subcluster summarizing a whole child node (sum of the node's CFs). The sums are
  # accumulated in subcluster order because scikit-learn builds them with repeated
  # `update` calls over the same order.
  defp parent_subcluster(node) do
    n = node.subclusters |> Enum.map(& &1.n) |> Enum.sum()
    ss = node.subclusters |> Enum.map(& &1.ss) |> Enum.sum()
    ls = node.subclusters |> Enum.map(& &1.ls) |> Enum.reduce(&Nx.add(&2, &1))
    centroid = Nx.divide(ls, n)

    %{n: n, ls: ls, ss: ss, centroid: centroid, sq_norm: squared_norm(centroid), child: node}
  end

  defp squared_norm(point), do: Nx.to_number(Nx.sum(Nx.multiply(point, point)))

  # ||c - x||^2 without the ||x||^2 term, which is constant across the subclusters of
  # a node. Dropping it is what scikit-learn does, and it has to be dropped here too:
  # adding it back rounds differently and would break ties against a different entry.
  defp closest_index(subclusters, point) do
    centroids = subclusters |> Enum.map(& &1.centroid) |> Nx.stack()
    squared_norms = subclusters |> Enum.map(& &1.sq_norm) |> Nx.tensor(type: Nx.type(centroids))

    centroids
    |> Nx.dot(point)
    |> Nx.multiply(-2)
    |> Nx.add(squared_norms)
    |> Nx.argmin()
    |> Nx.to_number()
  end

  defp leaf_centroids(%{is_leaf: true} = node), do: Enum.map(node.subclusters, & &1.centroid)

  defp leaf_centroids(node),
    do: Enum.flat_map(node.subclusters, fn sc -> leaf_centroids(sc.child) end)

  # --- Global clustering of the leaf subcluster centroids ---

  # Fewer subclusters than requested clusters means there is nothing to merge and
  # each subcluster becomes its own cluster.
  defp global_labels(subcluster_centers, key, opts) do
    num_subclusters = Nx.axis_size(subcluster_centers, 0)

    if num_subclusters <= opts[:num_clusters] do
      Nx.iota({num_subclusters}, type: :s32)
    else
      model =
        Scholar.Cluster.KMeans.fit(subcluster_centers,
          num_clusters: opts[:num_clusters],
          num_runs: opts[:num_runs],
          max_iterations: opts[:max_iterations],
          key: key
        )

      Nx.as_type(model.labels, :s32)
    end
  end

  defnp assign_labels(subcluster_centers, subcluster_labels, x) do
    nearest =
      Scholar.Metrics.Distance.pairwise_euclidean(x, subcluster_centers)
      |> Nx.argmin(axis: 1)

    Nx.take(subcluster_labels, nearest)
  end

  @doc """
  Makes predictions with the given `model` on inputs `x`.

  ## Return Values

  It returns a tensor with cluster labels corresponding to the input.

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> x = Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]])
      iex> model = Scholar.Cluster.BIRCH.fit(x, num_clusters: 2, key: key)
      iex> Scholar.Cluster.BIRCH.predict(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
      #Nx.Tensor<
        s32[2]
        [1, 0]
      >
  """
  defn predict(
         %__MODULE__{subcluster_centers: subcluster_centers, subcluster_labels: subcluster_labels} =
           _model,
         x
       ) do
    assert_same_shape!(x[0], subcluster_centers[0])

    assign_labels(subcluster_centers, subcluster_labels, x)
  end
end
