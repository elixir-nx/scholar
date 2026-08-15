defmodule Scholar.Cluster.HDBSCAN do
  @moduledoc """
  HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications with Noise.

  HDBSCAN turns DBSCAN into a hierarchical algorithm and then extracts a flat clustering
  from that hierarchy. Unlike `Scholar.Cluster.DBSCAN` it does not need an `eps` radius,
  so it can recover clusters of differing densities, and points that belong to no cluster
  are labelled as noise.

  The algorithm proceeds in four steps:

    1. Compute the core distance of each point, the distance to its `:min_samples`-th
       neighbour, which is a local estimate of how sparse the point's neighbourhood is.

    2. Build the mutual reachability, $d_{mreach}(a, b) = \\max(core(a), core(b), d(a, b))$,
       which pushes sparse points away from everything while leaving dense regions untouched.

    3. Build a single linkage hierarchy over the mutual reachability, via
       `Scholar.Cluster.Hierarchical`.

    4. Condense that hierarchy using `:min_cluster_size` and select the flat clustering
       that maximises cluster stability (the "excess of mass" criterion).

  The time complexity is $O(N^2 \\log N)$ for $N$ samples. The space complexity is $O(N^2)$,
  since the mutual reachability is held as a dense matrix.

  ## Not yet supported

  Compared to `sklearn.cluster.HDBSCAN`, the following are not implemented yet:

    * `cluster_selection_method: :leaf`, which selects leaf clusters instead of applying
      the excess of mass criterion.
    * `cluster_selection_epsilon`, which merges clusters below a distance threshold.
    * `max_cluster_size`.
    * `allow_single_cluster`, so the root is never returned as a cluster and a dataset with
      no real split comes back as all noise.
    * Membership probabilities.

  ## References

    * [Density-Based Clustering Based on Hierarchical Density Estimates](https://doi.org/10.1007/978-3-642-37456-2_14),
      Campello, Moulavi and Sander, 2013.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:labels]}
  defstruct [:labels]

  opts = [
    min_cluster_size: [
      type: :pos_integer,
      default: 5,
      doc: """
      The smallest group of points that should be considered a cluster. Groups smaller than
      this are treated as points falling out of their parent cluster. Must be at least `2`.
      """
    ],
    min_samples: [
      type: :pos_integer,
      doc: """
      The number of samples in a neighbourhood for a point to be considered a core point,
      including the point itself. Larger values make the clustering more conservative, so
      more points end up as noise. Defaults to `:min_cluster_size`.
      """
    ],
    metric: [
      type: {:custom, Scholar.Neighbors.Utils, :pairwise_metric, []},
      default: &Scholar.Metrics.Distance.pairwise_euclidean/2,
      doc: ~S"""
      The function that measures the pairwise distance between two points. Possible values:

      * `{:minkowski, p}` - Minkowski metric. By changing value of `p` parameter (a positive number or `:infinity`)
      we can set Manhattan (`1`), Euclidean (`2`), Chebyshev (`:infinity`), or any arbitrary $L_p$ metric.

      * `:euclidean` - Euclidean metric.

      * `:squared_euclidean` - Squared Euclidean metric.

      * `:manhattan` - Manhattan metric.

      * `:cosine` - Cosine metric.

      * Anonymous function of arity 2 that takes two rank-2 tensors.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Performs HDBSCAN clustering on `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

  * `:labels` - Cluster labels for each point in `x`. Noisy samples are given the label `-1`.
    Clusters are numbered from `0`, ordered by the point at which they appear in the
    condensed hierarchy.

  ## Examples

      iex> x = Nx.tensor([[1, 2], [2, 2], [2, 3], [1, 3], [8, 7], [8, 8], [7, 8], [7, 7]])
      iex> model = Scholar.Cluster.HDBSCAN.fit(x, min_cluster_size: 3, min_samples: 2)
      iex> model.labels
      #Nx.Tensor<
        s32[8]
        [0, 0, 0, 0, 1, 1, 1, 1]
      >
  """
  deftransform fit(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {num_samples, num_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    min_cluster_size = opts[:min_cluster_size]

    if min_cluster_size < 2 do
      raise ArgumentError,
            "expected :min_cluster_size to be at least 2, got: #{min_cluster_size}"
    end

    num_samples = Nx.axis_size(x, 0)

    if num_samples < 3 do
      raise ArgumentError,
            "expected x to have at least 3 samples, got: #{num_samples}"
    end

    min_samples = opts[:min_samples] || min_cluster_size

    if min_samples > num_samples do
      raise ArgumentError,
            "expected :min_samples to be at most the number of samples (#{num_samples}), got: #{min_samples}"
    end

    fit_n(x,
      min_cluster_size: min_cluster_size,
      min_samples: min_samples,
      metric: opts[:metric]
    )
  end

  defnp fit_n(x, opts) do
    mutual_reachability = mutual_reachability(x, opts)

    model =
      Scholar.Cluster.Hierarchical.fit(mutual_reachability,
        dissimilarity: :precomputed,
        linkage: :single
      )

    labels =
      extract_clusters(
        model.clades,
        model.dissimilarities,
        model.sizes,
        min_cluster_size: opts[:min_cluster_size]
      )

    %__MODULE__{labels: labels}
  end

  # d_mreach(a, b) = max(core(a), core(b), d(a, b)), where core(a) is the distance from a
  # to its min_samples-th neighbour counting itself.
  defnp mutual_reachability(x, opts) do
    metric = opts[:metric]
    min_samples = opts[:min_samples]

    distances = metric.(x, x)
    core = Nx.sort(distances, axis: 1)[[.., min_samples - 1]]

    Nx.max(Nx.max(Nx.new_axis(core, 1), Nx.new_axis(core, 0)), distances)
  end

  # Condenses the single linkage hierarchy and selects clusters by excess of mass.
  #
  # A condensed cluster is identified by the node at which it was born, so cluster ids live
  # in the same 0..2n-2 space as tree nodes. A cluster born at node `c` always has a smaller
  # id than the cluster containing it, which makes ascending id a bottom-up traversal and
  # descending id a top-down one. Every pass below is therefore a `while` with a static trip
  # count over preallocated tensors, with no recursion and no data dependent control flow.
  defnp extract_clusters(clades, dissimilarities, sizes, opts) do
    min_cluster_size = opts[:min_cluster_size]
    {num_merges, _} = Nx.shape(clades)
    n = num_merges + 1
    total = 2 * n - 1
    root = total - 1

    node_cluster = Nx.broadcast(Nx.s64(-1), {total})
    node_cluster = Nx.put_slice(node_cluster, [root], Nx.reshape(Nx.s64(root), {1}))
    node_fallen = Nx.broadcast(Nx.u8(0), {total})
    node_lambda = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(dissimilarities)), {total})

    is_cluster = Nx.broadcast(Nx.u8(0), {total})
    is_cluster = Nx.put_slice(is_cluster, [root], Nx.reshape(Nx.u8(1), {1}))
    cluster_parent = Nx.broadcast(Nx.s64(-1), {total})
    cluster_birth = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(dissimilarities)), {total})
    stability = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(dissimilarities)), {total})

    # Clusters are numbered by the order in which they are born during the top-down pass.
    # `birth_order` maps a cluster to its slot, `birth_cluster` maps a slot back to a cluster.
    # Two clusters are born per split, so there are at most 2 * (n - 1) slots, plus the root.
    max_births = 2 * n
    birth_order = Nx.broadcast(Nx.s64(0), {total})
    birth_cluster = Nx.broadcast(Nx.s64(-1), {max_births})

    # Pass 1, top-down: condense the hierarchy and accumulate the stability that each
    # cluster gains from sub-clusters splitting off.
    {node_cluster, node_fallen, node_lambda, is_cluster, cluster_parent, cluster_birth, stability,
     birth_order, birth_cluster, _, _} =
      while {node_cluster, node_fallen, node_lambda, is_cluster, cluster_parent, cluster_birth,
             stability, birth_order, birth_cluster, births = Nx.s64(0),
             {clades, dissimilarities, sizes, step = Nx.u32(0)}},
            step < num_merges do
        i = num_merges - 1 - Nx.as_type(step, :s64)
        k = n + i
        left = Nx.as_type(clades[[i, 0]], :s64)
        right = Nx.as_type(clades[[i, 1]], :s64)
        # Duplicate points merge at distance 0, so lambda is infinite there. Both sides of
        # Nx.select are evaluated, and dividing by zero raises on some backends, so the
        # denominator is made safe before the division rather than after it.
        dist = dissimilarities[i]
        safe_dist = Nx.select(dist > 0, dist, 1)

        lambda =
          Nx.select(dist > 0, 1 / safe_dist, Nx.Constants.infinity(Nx.type(dissimilarities)))

        left_size = subtree_size(sizes, left, n)
        right_size = subtree_size(sizes, right, n)
        left_big = left_size >= min_cluster_size
        right_big = right_size >= min_cluster_size

        fallen? = node_fallen[k] == 1
        current = node_cluster[k]
        both_big = left_big and right_big
        neither_big = not left_big and not right_big

        {left_cluster, left_fallen, left_lambda, left_born} =
          child_fate(
            left,
            current,
            fallen?,
            node_lambda[k],
            lambda,
            both_big,
            neither_big,
            left_big
          )

        {right_cluster, right_fallen, right_lambda, right_born} =
          child_fate(
            right,
            current,
            fallen?,
            node_lambda[k],
            lambda,
            both_big,
            neither_big,
            right_big
          )

        children = Nx.stack([left, right]) |> Nx.new_axis(-1)

        node_cluster =
          Nx.indexed_put(node_cluster, children, Nx.stack([left_cluster, right_cluster]))

        node_fallen = Nx.indexed_put(node_fallen, children, Nx.stack([left_fallen, right_fallen]))
        node_lambda = Nx.indexed_put(node_lambda, children, Nx.stack([left_lambda, right_lambda]))

        # Register the newly born clusters. Writing back the existing value keeps the
        # scatter branch free when no cluster was born.
        is_cluster =
          Nx.indexed_put(
            is_cluster,
            children,
            Nx.stack([
              Nx.select(left_born, Nx.u8(1), is_cluster[left]),
              Nx.select(right_born, Nx.u8(1), is_cluster[right])
            ])
          )

        cluster_parent =
          Nx.indexed_put(
            cluster_parent,
            children,
            Nx.stack([
              Nx.select(left_born, current, cluster_parent[left]),
              Nx.select(right_born, current, cluster_parent[right])
            ])
          )

        cluster_birth =
          Nx.indexed_put(
            cluster_birth,
            children,
            Nx.stack([
              Nx.select(left_born, lambda, cluster_birth[left]),
              Nx.select(right_born, lambda, cluster_birth[right])
            ])
          )

        # A cluster's stability grows by (lambda - its own birth lambda) for every point
        # that leaves it, and a splitting sub-cluster takes all of its points at once.
        # `current` is always an ancestor of both children, so it never aliases the writes above.
        birth = cluster_birth[current]

        gained =
          Nx.select(left_born, (lambda - birth) * left_size, 0) +
            Nx.select(right_born, (lambda - birth) * right_size, 0)

        stability =
          Nx.indexed_add(
            stability,
            Nx.reshape(Nx.max(current, 0), {1, 1}),
            Nx.reshape(Nx.select(current >= 0, gained, 0), {1})
          )

        # A split births the left child first, then the right one, matching the order a
        # top-down walk of the hierarchy would visit them.
        left_slot = births
        right_slot = births + 1

        birth_order =
          Nx.indexed_put(
            birth_order,
            children,
            Nx.stack([
              Nx.select(left_born, left_slot, birth_order[left]),
              Nx.select(right_born, right_slot, birth_order[right])
            ])
          )

        birth_cluster =
          Nx.indexed_put(
            birth_cluster,
            Nx.stack([left_slot, right_slot]) |> Nx.new_axis(-1),
            Nx.stack([
              Nx.select(left_born, left, birth_cluster[left_slot]),
              Nx.select(right_born, right, birth_cluster[right_slot])
            ])
          )

        births = births + Nx.select(left_born, 2, 0)

        {node_cluster, node_fallen, node_lambda, is_cluster, cluster_parent, cluster_birth,
         stability, birth_order, birth_cluster, births,
         {clades, dissimilarities, sizes, step + 1}}
      end

    # Pass 2: individual points that dropped out contribute to their cluster's stability.
    {stability, _} =
      while {stability, {node_cluster, node_fallen, node_lambda, cluster_birth, p = Nx.u32(0)}},
            p < n do
        idx = Nx.as_type(p, :s64)
        cluster = node_cluster[idx]
        counts? = cluster >= 0 and node_fallen[idx] == 1
        gained = Nx.select(counts?, node_lambda[idx] - cluster_birth[Nx.max(cluster, 0)], 0)

        stability =
          Nx.indexed_add(
            stability,
            Nx.reshape(Nx.max(cluster, 0), {1, 1}),
            Nx.reshape(gained, {1})
          )

        {stability, {node_cluster, node_fallen, node_lambda, cluster_birth, p + 1}}
      end

    # Pass 3, bottom-up: a cluster is kept only if it is more stable than its descendants
    # taken together. Ascending id visits children before parents.
    selected = is_cluster
    selected = Nx.put_slice(selected, [root], Nx.reshape(Nx.u8(0), {1}))

    {selected, _stability, _} =
      while {selected, stability,
             {is_cluster, cluster_parent,
              subtree = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(stability)), {total}),
              step = Nx.u32(0)}},
            step < total - n do
        c = n + Nx.as_type(step, :s64)
        exists? = is_cluster[c] == 1
        beaten? = exists? and c != root and subtree[c] > stability[c]

        selected =
          Nx.indexed_put(
            selected,
            Nx.reshape(c, {1, 1}),
            Nx.reshape(Nx.select(beaten?, Nx.u8(0), selected[c]), {1})
          )

        stability =
          Nx.indexed_put(
            stability,
            Nx.reshape(c, {1, 1}),
            Nx.reshape(Nx.select(beaten?, subtree[c], stability[c]), {1})
          )

        parent = cluster_parent[c]

        subtree =
          Nx.indexed_add(
            subtree,
            Nx.reshape(Nx.max(parent, 0), {1, 1}),
            Nx.reshape(Nx.select(exists? and parent >= 0, stability[c], 0), {1})
          )

        {selected, stability, {is_cluster, cluster_parent, subtree, step + 1}}
      end

    # Pass 4, top-down: once a cluster is selected none of its descendants may be, and each
    # cluster resolves to the selected ancestor that will own its points.
    resolved = Nx.broadcast(Nx.s64(-1), {total})

    resolved =
      Nx.put_slice(
        resolved,
        [root],
        Nx.reshape(Nx.select(selected[root] == 1, Nx.s64(root), Nx.s64(-1)), {1})
      )

    {selected, resolved, _} =
      while {selected, resolved, {cluster_parent, step = Nx.u32(0)}}, step < total - n do
        c = root - 1 - Nx.as_type(step, :s64)
        parent = cluster_parent[c]
        parent_owns? = parent >= 0 and resolved[Nx.max(parent, 0)] >= 0
        keep? = selected[c] == 1 and not parent_owns?

        selected =
          Nx.indexed_put(
            selected,
            Nx.reshape(c, {1, 1}),
            Nx.reshape(Nx.select(keep?, Nx.u8(1), Nx.u8(0)), {1})
          )

        owner =
          Nx.select(
            keep?,
            c,
            Nx.select(parent >= 0, resolved[Nx.max(parent, 0)], Nx.s64(-1))
          )

        resolved = Nx.indexed_put(resolved, Nx.reshape(c, {1, 1}), Nx.reshape(owner, {1}))

        {selected, resolved, {cluster_parent, step + 1}}
      end

    # Number the surviving clusters from 0 in the order they were born, walking the hierarchy
    # from the root down, then label every point by the cluster that owns it.
    born_selected = Nx.take(selected, Nx.max(birth_cluster, 0))
    born_selected = Nx.select(birth_cluster >= 0, born_selected, Nx.u8(0))
    rank_by_birth = Nx.cumulative_sum(Nx.as_type(born_selected, :s64)) - 1

    point_cluster = Nx.take(node_cluster, Nx.iota({n}))
    owner = Nx.take(resolved, Nx.max(point_cluster, 0))
    owner = Nx.select(point_cluster >= 0, owner, Nx.s64(-1))

    owner_birth = Nx.take(birth_order, Nx.max(owner, 0))
    labels = Nx.take(rank_by_birth, Nx.max(owner_birth, 0))
    Nx.select(owner >= 0, labels, Nx.s64(-1)) |> Nx.as_type(:s32)
  end

  # Number of original points under `node`: 1 for a leaf, else the size recorded for the
  # merge that created it.
  defnp subtree_size(sizes, node, n) do
    Nx.select(node < n, 1, Nx.take(sizes, Nx.max(node - n, 0)))
  end

  # Resolves what happens to one child of the node being processed.
  #
  #   * the parent already fell out: the child inherits its cluster and drop lambda
  #   * both children are big enough: this is a real split, the child starts a new cluster
  #   * neither is big enough: the cluster dies here and the child drops out
  #   * exactly one is big enough: the big one continues the parent cluster, the small one drops
  defnp child_fate(child, current, fallen?, parent_lambda, lambda, both_big, neither_big, big?) do
    cluster = Nx.select(fallen?, current, Nx.select(both_big, child, current))

    fell =
      Nx.select(
        fallen?,
        Nx.u8(1),
        Nx.select(
          both_big,
          Nx.u8(0),
          Nx.select(neither_big, Nx.u8(1), Nx.select(big?, Nx.u8(0), Nx.u8(1)))
        )
      )

    drop_lambda = Nx.select(fallen?, parent_lambda, lambda)
    born = not fallen? and both_big

    {cluster, fell, drop_lambda, born}
  end
end
