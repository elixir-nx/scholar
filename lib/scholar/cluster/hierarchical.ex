defmodule Scholar.Cluster.Hierarchical do
  @moduledoc """
  Performs [hierarchical, agglomerative clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering#Agglomerative_clustering_example)
  on a dataset.

  Hierarchical clustering is good for when the number of clusters is not known ahead of time.
  It also allows for the creation of a [dendrogram plot](https://en.wikipedia.org/wiki/Dendrogram)
  (regardless of the dimensionality of the dataset) which can be used to select the number of
  clusters in a post-processing step.

  ## Limitations

  Due to the requirements of the current implementation, only these options are supported:

    * `dissimilarity: :euclidean | :precomputed`
    * `linkage: :average | :complete | :single | :ward | :weighted`

  We use the nearest-neighbor-chain algorithm, which is $O(n^2)$ in both time and memory, where
  $n$ is the number of data points.
  This is better than the generic algorithm, which is $O(n^3)$, and is the best known bound for
  this family of linkages.

  However, the implementation requires certain theoretical properties of the dissimilarities and
  linkages.
  As such, we've restricted the options to only those combinations with the correct properties.

  In the future, we plan to add additional algorithms which won't have the same restrictions.
  """
  import Nx.Defn

  @derive {Nx.Container, keep: [:num_points], containers: [:clades, :dissimilarities, :sizes]}
  defstruct [:clades, :dissimilarities, :num_points, :sizes]

  @dissimilarity_types [
    :euclidean,
    :precomputed
  ]

  @linkage_types [
    :average,
    # :centroid,
    :complete,
    # :median,
    :single,
    :ward,
    :weighted
  ]

  @fit_opts_schema [
    dissimilarity: [
      type: {:in, @dissimilarity_types},
      default: :euclidean,
      doc: """
      Pairwise dissimilarity function: computes the 'dissimilarity' between each pair of data points.
      Dissimilarity is analogous to distance, but without the expectation that the triangle
      inequality holds.

      Choices:

        * `:euclidean` - L2 norm.

        * `:precomputed` - `data` is already a pairwise dissimilarity matrix of shape
          `{num_obs, num_obs}` and is used as is. It must be symmetric. Its diagonal is
          ignored, so it may hold any value. This is useful when the dissimilarity is
          expensive to recompute, comes from outside Scholar, or is not derived from
          coordinates at all, such as the mutual reachability used by density based
          clustering.

          Symmetry is not checked, since that would require inspecting the tensor.
          Note also that `linkage: :ward` assumes euclidean dissimilarities, so pairing
          it with an arbitrary precomputed matrix is not meaningful.

      See "Limitations" in the moduledoc for an explanation of the lack of choices.
      """
    ],
    linkage: [
      type: {:in, @linkage_types},
      default: :single,
      doc: ~S"""
      Linkage function: how to compute the intra-clade dissimilarity of two clades if they were
      merged.

      Choices:

        * `:average` - The unweighted average dissimilarity across all pairs of points.

        * `:complete` - (Historic name) The maximum dissimilarity across all pairs of points.

        * `:single` - (Historic name) The minimum dissimilarity across all pairs of points.

        * `:ward` - (Named for [Ward's method](https://en.wikipedia.org/wiki/Ward%27s_method))
          The minimum increase in sum of squares (MISSQ) of dissimilarities.

        * `:weighted` - The weighted average dissimilarity across all pairs of points.
      """
    ]
  ]
  @doc """
  Use hierarchical clustering to form the initial model to be clustered with `labels_list/2` or
  `labels_map/2`.

  ## Options

  #{NimbleOptions.docs(@fit_opts_schema)}

  ## Return values

  Returns a `Scholar.Cluster.Hierarchical` struct with the following fields:

    * `clades` (`Nx.Tensor` with shape `{n - 1, 2}`) -
      Contains the indices of the pair of clades merged at each step of the agglomerative
      clustering process.

      Agglomerative clustering starts by considering each datum in `data` its own singleton group
      or ["clade"](https://en.wikipedia.org/wiki/Clade).
      It then picks two clades to merge into a new clade containing the data from both.
      It does this until there is a single clade remaining.

      Since each datum starts as its own clade, e.g. `data[0]` is clade `0`, indexing of new clades
      starts at `n` where `n` is the size of the original `data` tensor.
      If `clades[k] == [i, j]`, then clades `i` and `j` were merged to form `k + n`.

    * `dissimilarities` (`Nx.Tensor` with shape `{n - 1}`) -
      Contains a metric that measures the intra-clade closeness of each newly formed clade.
      Represented by the heights of each clade in a dendrogram plot.
      Determined by both the `:dissimilarity` and `:linkage` options.

    * `num_points` (`pos_integer/0`) -
      Number of points in the dataset.
      Must be $\\geq 3$.

    * `sizes` (`Nx.Tensor` with shape `{n - 1}`) -
      `sizes[i]` is the size of clade `i`.
      If clade `k` was created by merging clades `i` and `j`, then
      `sizes[k] == sizes[i] + sizes[j]`.

  ## Incomplete dendrograms

  Non-finite dissimilarities, which come from `:nan` or infinite values in `data` or in a
  precomputed matrix, can leave two or more clades with no finite distance to merge by. There
  is then no meaningful pair to merge next, so the remaining merges are not made and are
  reported as `clades` of `[-1, -1]`, `sizes` of `0`, and `NaN` dissimilarities, sorted to the
  end. Test for them with `clades` or `sizes`, since a merge that really was made can also
  carry a `NaN` dissimilarity when the underlying distances are themselves `NaN`:

      Nx.any(Nx.equal(model.sizes, 0))

  ## Examples

      iex> data = Nx.tensor([[2], [7], [9], [0], [3]])
      iex> Hierarchical.fit(data)
      %Scholar.Cluster.Hierarchical{
        clades: Nx.tensor([[0, 4], [3, 5], [1, 2], [6, 7]]),
        dissimilarities: Nx.tensor([1.0, 2.0, 2.0, 4.0]),
        num_points: 5,
        sizes: Nx.tensor([2, 3, 2, 5])
      }

  Passing the pairwise dissimilarities directly gives the same model:

      iex> data = Nx.tensor([[2], [7], [9], [0], [3]])
      iex> dissimilarities = Scholar.Metrics.Distance.pairwise_euclidean(data)
      iex> Hierarchical.fit(dissimilarities, dissimilarity: :precomputed)
      %Scholar.Cluster.Hierarchical{
        clades: Nx.tensor([[0, 4], [3, 5], [1, 2], [6, 7]]),
        dissimilarities: Nx.tensor([1.0, 2.0, 2.0, 4.0]),
        num_points: 5,
        sizes: Nx.tensor([2, 3, 2, 5])
      }
  """
  deftransform fit(%Nx.Tensor{} = data, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @fit_opts_schema)
    dissimilarity = opts[:dissimilarity]
    linkage = opts[:linkage]

    dissimilarity_fun =
      case dissimilarity do
        :precomputed -> &Function.identity/1
        :euclidean -> &Scholar.Metrics.Distance.pairwise_euclidean/1
      end

    update_fun =
      case linkage do
        :average -> &average/6
        # :centroid -> &centroid/6
        :complete -> &complete/6
        # :median -> &median/6
        :single -> &single/6
        :ward -> &ward/6
        :weighted -> &weighted/6
      end

    dendrogram_fun =
      case linkage do
        # TODO: :centroid, :median
        l when l in [:average, :complete, :single, :ward, :weighted] ->
          &nn_chain/3
      end

    n =
      case {dissimilarity, Nx.shape(data)} do
        {:precomputed, {n, n}} ->
          n

        {:precomputed, other} ->
          raise ArgumentError,
                "Expected a square rank 2 (`{num_obs, num_obs}`) tensor when `dissimilarity: :precomputed`, found shape: #{inspect(other)}."

        {_, {n, _num_features}} ->
          n

        {_, other} ->
          raise ArgumentError,
                "Expected a rank 2 (`{num_obs, num_features}`) tensor, found shape: #{inspect(other)}."
      end

    if n < 3 do
      raise ArgumentError, "Must have a minimum of 3 data points, found: #{n}."
    end

    {clades, diss, sizes} = dendrogram_fun.(data, dissimilarity_fun, update_fun)

    %__MODULE__{
      clades: clades,
      dissimilarities: diss,
      num_points: n,
      sizes: sizes
    }
  end

  # Clade functions

  # Nearest-neighbor-chain algorithm (Muellner, "Modern hierarchical, agglomerative
  # clustering algorithms", 2011), the one SciPy uses for this same family of
  # Lance-Williams-updatable linkages. Builds a chain of mutually-adjacent nearest
  # neighbors (point -> its nearest -> that point's nearest -> ...), which is
  # mathematically guaranteed to terminate in a pair that are each other's nearest
  # neighbor (a chain can only ever cycle back on its immediate predecessor: three
  # points a -> b -> c -> a would force d(c,a) < d(b,c) < d(a,b) <= d(a,c) = d(c,a),
  # a contradiction). That pair merges, gets popped off the chain, and the chain
  # continues extending from whatever is left, rather than restarting from scratch.
  #
  # Extending the chain by one point costs O(n) (a single row of the pairwise
  # matrix). The total number of extensions over the whole run is O(n): a merge
  # shortens the chain by 2, a non-terminating extension lengthens it by 1, and
  # the chain can never exceed n, so extensions are amortized O(1) per merge. That
  # makes the whole algorithm O(n^2) total, without relying on many clades being
  # simultaneously mutual nearest neighbors of each other, which a round-based
  # parallel merge needs for its own complexity to hold. That assumption stops
  # holding once few enough clades remain: nearest-neighbor relationships form
  # long chains rather than isolated mutual pairs, which is normal, not a bug in
  # the data, but it means a round-based approach degrades to one merge per
  # round for however many clades are left, each round still paying to scan the
  # entire matrix. That is O(n) such rounds at O(n^2) each, i.e. O(n^3) overall,
  # which is what this algorithm avoids.
  #
  # Reaching that O(n^2) in practice also depends on the pairwise matrix being
  # updated in place across loop iterations. Copying it even once per merge would
  # put the O(n^3) right back. Two things defeat the in-place update, both of them
  # invisible in the source and both worth preserving as written below: reading a
  # single cell of the matrix into another loop variable (which is why the merge
  # distance is carried out of the chain loop instead of read back from the
  # matrix), and writing the merged clade's row before its column rather than
  # after.
  defnp nn_chain(data, dissimilarity_fun, update_fun) do
    pairwise = dissimilarity_fun.(data)
    {n, _} = Nx.shape(pairwise)
    pairwise = Nx.broadcast(:infinity, {n}) |> Nx.make_diagonal() |> Nx.add(pairwise)
    clades = Nx.broadcast(-1, {n - 1, 2})
    sizes = Nx.broadcast(1, {2 * n - 1})
    # Slot -> id of the clade currently living in it. Slots start out holding the
    # singleton clades, and a merge overwrites the surviving slot with the new id.
    pointers = Nx.iota({n})

    cluster_sizes = Nx.broadcast(1, {n})
    diss = Nx.tensor(:infinity, type: Nx.type(pairwise)) |> Nx.broadcast({n - 1})
    alive = Nx.broadcast(Nx.u8(1), {n})
    chain = Nx.broadcast(-1, {n})

    {{clades, diss, sizes, _cluster_sizes, count}, _} =
      while {{clades, diss, sizes, cluster_sizes, count = 0},
             {pointers, pairwise, alive, chain, chain_length = 0, aborted = Nx.u8(0)}},
            count < n - 1 and aborted == 0 do
        # Start a new chain from any live clade if the previous merge emptied it.
        needs_start = chain_length == 0
        start = Nx.argmax(alive)

        chain =
          Nx.select(
            needs_start,
            Nx.indexed_put(chain, Nx.reshape(0, {1, 1}), Nx.reshape(start, {1})),
            chain
          )

        chain_length = Nx.select(needs_start, 1, chain_length)

        # Extend the chain until its last two entries are mutual nearest neighbors.
        # Bounded by n: the distance along a chain strictly decreases, so a chain
        # can never revisit a clade before terminating, provided ties end it rather
        # than extend it (see `previous_is_nearest` below). Exceeding n extensions
        # then only happens with a non-finite dissimilarity (from NaN or infinite
        # input), where argmin's tie-breaking can land on an already-dead clade and
        # loop without ever finding a genuine mutual pair.
        {chain, chain_length, found, _steps, merge_diss, _pairwise, _alive} =
          while {chain, chain_length, found = Nx.u8(0), steps = 0,
                 _chain_diss = Nx.Constants.infinity(Nx.type(pairwise)), pairwise, alive},
                found == 0 and steps <= n do
            tip = chain[chain_length - 1]
            previous = Nx.select(chain_length > 1, chain[Nx.max(chain_length - 2, 0)], -1)

            # Merged-away clades are masked out here rather than overwritten in the
            # matrix: this costs O(n) per step, whereas blanking their row and
            # column would cost a full n x n rewrite on every merge.
            row = Nx.select(alive, pairwise[tip], Nx.Constants.infinity(Nx.type(pairwise)))
            nearest = Nx.argmin(row)

            # Distance to that neighbor, carried out so that the merge below does
            # not have to read it back out of the matrix. That read looks free but
            # is not: feeding a cell of the matrix into another loop variable stops
            # XLA from updating the matrix in place, costing a full n x n copy on
            # every merge. See the note above nn_chain/3.
            chain_diss = Nx.reduce_min(row)

            # A point is never its own nearest neighbor, but argmin cannot tell the
            # difference once a row is entirely non-finite: the diagonal is always
            # infinity by construction, same as every dead or genuinely
            # infinitely-far entry, so a tie-break can land right back on tip. That
            # is only possible with a non-finite dissimilarity (see the module
            # comment above); treat it as stalled rather than let the chain merge a
            # clade with itself.
            self_match = nearest == tip

            # The chain terminates whenever the previous entry is *a* nearest
            # neighbor of the tip, not only when it is the one argmin happens to
            # return. The two differ exactly when the tip's nearest distance is
            # tied, and preferring the previous entry there is what keeps the
            # chain from walking back onto a clade it already holds: a duplicated
            # entry survives the merge that pops its other occurrence, and is then
            # merged a second time, after it is already gone. The chain can only
            # cycle back on its immediate predecessor while distances strictly
            # decrease, which ties break.
            previous_is_nearest =
              chain_length > 1 and
                Nx.take(row, Nx.max(previous, 0)) == chain_diss

            # `nearest == previous` on its own would miss a tie, and comparing the
            # distances on its own would miss a NaN, which is never equal to
            # itself. Either one ending the chain is enough.
            found = (previous_is_nearest or nearest == previous) and not self_match

            # The chain holds live clades and so cannot outgrow its buffer, but a
            # non-finite dissimilarity can stall it without ever finding a pair, and
            # a write past the end would otherwise be silently clamped onto the last
            # slot. Treat it as stalled, like a self match.
            stalled = self_match or chain_length >= n

            chain =
              Nx.select(
                found or stalled,
                chain,
                Nx.indexed_put(
                  chain,
                  Nx.reshape(Nx.min(chain_length, n - 1), {1, 1}),
                  Nx.reshape(nearest, {1})
                )
              )

            chain_length = Nx.select(found or stalled, chain_length, chain_length + 1)
            steps = Nx.select(stalled, n + 1, steps + 1)
            {chain, chain_length, found, steps, chain_diss, pairwise, alive}
          end

        i = Nx.max(Nx.min(chain[chain_length - 1], chain[Nx.max(chain_length - 2, 0)]), 0)
        j = Nx.max(Nx.max(chain[chain_length - 1], chain[Nx.max(chain_length - 2, 0)]), 0)

        {clades, count, pointers, pairwise, diss, sizes, cluster_sizes, alive} =
          merge_one(
            clades,
            count,
            pointers,
            pairwise,
            diss,
            sizes,
            cluster_sizes,
            alive,
            i,
            j,
            found,
            merge_diss,
            n,
            update_fun
          )

        chain_length = Nx.select(found, chain_length - 2, chain_length)

        # A non-finite dissimilarity is the only way the chain extension above can
        # exhaust its step budget without finding a mutual pair. Stop instead of
        # looping forever or guessing a merge that has no finite distance to
        # justify it: see the comment on the incomplete-rows handling below.
        aborted = found == 0

        {{clades, diss, sizes, cluster_sizes, count},
         {pointers, pairwise, alive, chain, chain_length, aborted}}
      end

    sizes = sizes[n..(2 * n - 2)]

    # Rows the loop never got to fill, if it aborted above. Marking their dissimilarity
    # NaN keeps them out of the way of the sort below (NaN orders after every real value,
    # including infinity) and reports the incomplete merges as such instead of leaving
    # their initial values looking like real ones.
    incomplete = Nx.iota({n - 1}) >= count
    diss = Nx.select(incomplete, Nx.Constants.nan(Nx.type(diss)), diss)

    perm = Nx.argsort(diss, stable: true, type: :u32)

    # A row at index `i` creates clade `n + i`. Reordering the rows therefore also requires
    # reordering every reference to a non-singleton clade.
    inverse_perm =
      Nx.broadcast(0, {n - 1})
      |> Nx.indexed_put(Nx.new_axis(perm, -1), Nx.iota({n - 1}, type: Nx.type(clades)))

    clade_id_mapping =
      Nx.concatenate([Nx.iota({n}, type: Nx.type(clades)), inverse_perm + n])

    # Incomplete rows sort to the tail, so the mask still marks them after the permutation.
    # Their clade ids stay -1 rather than being mapped through the table.
    sorted_clades = clades[perm]

    # Each row already holds [min, max] in terms of the ids merge_one saw, but remapping
    # ids to their final, sorted-by-dissimilarity numbering does not preserve that order when
    # two rows are tied: the mapping is a permutation of ids, not a monotonic one on ties. Sort
    # again after remapping so a row is always [min, max] in the ids callers actually see.
    clades =
      Nx.select(
        Nx.broadcast(Nx.new_axis(incomplete, -1), Nx.shape(sorted_clades)),
        -1,
        Nx.take(clade_id_mapping, Nx.max(sorted_clades, 0)) |> Nx.sort(axis: 1)
      )

    sizes = Nx.select(incomplete, 0, sizes[perm])

    {clades, diss[perm], sizes}
  end

  # Merges clades i and j (or does nothing if `found` is false, which only happens when
  # the chain above stalled on a non-finite dissimilarity). Always does exactly the O(n)
  # work of one merge, rather than scanning every one of the n possible pairs to find the
  # (at most one) real one: nn_chain only ever has one pair to merge per call, unlike the
  # round-based approach this replaced, which could have many simultaneously and needed
  # that scan.
  defnp merge_one(
          clades,
          count,
          pointers,
          pairwise,
          diss,
          sizes,
          cluster_sizes,
          alive,
          i,
          j,
          found,
          merge_diss,
          n,
          update_fun
        ) do
    indices = [i, j] |> Nx.stack() |> Nx.new_axis(-1)
    a = pointers[i]
    b = pointers[j]
    c = count + n

    new_clade = Nx.stack([a, b]) |> Nx.sort() |> Nx.new_axis(0)
    clades = Nx.select(found, Nx.put_slice(clades, [count, 0], new_clade), clades)

    sa = sizes[i]
    sb = sizes[j]
    sc = sa + sb

    sizes =
      Nx.select(
        found,
        Nx.indexed_put(sizes, Nx.stack([i, c]) |> Nx.new_axis(-1), Nx.stack([sc, sc])),
        sizes
      )

    cluster_sizes =
      Nx.select(
        found,
        Nx.indexed_put(cluster_sizes, Nx.stack([i, j]) |> Nx.new_axis(-1), Nx.stack([sc, sc])),
        cluster_sizes
      )

    diss = Nx.select(found, Nx.indexed_put(diss, Nx.stack([count]), merge_diss), diss)

    # Only i survives the merge, so only its slot has to point at the new clade.
    pointers =
      Nx.select(
        found,
        Nx.indexed_put(pointers, Nx.reshape(i, {1, 1}), Nx.reshape(c, {1})),
        pointers
      )

    # j is no longer a live clade; i now stands for the merged one.
    alive_after_j_dies =
      Nx.indexed_put(
        alive,
        Nx.reshape(j, {1, 1}),
        Nx.broadcast(0, {1}) |> Nx.as_type(Nx.type(alive))
      )

    alive = Nx.select(found, alive_after_j_dies, alive)

    updates =
      update_fun.(pairwise[i], pairwise[j], merge_diss, sa, sb, cluster_sizes)
      |> Nx.indexed_put(indices, Nx.broadcast(:infinity, {2}))

    # Only the surviving clade's row and column are rewritten; j's are left stale
    # and masked out through `alive` when the chain reads a row. Unguarded on
    # purpose: when there is no pair to merge the caller aborts on this same
    # iteration and discards `pairwise`, so writing to it is harmless, whereas
    # guarding it with Nx.select would force a full n x n copy on every merge.
    # Column first, then row. The order is not cosmetic: writing the row first
    # and the column second makes XLA fall back to copying the whole matrix on
    # every merge, which is O(n^3) overall. See the comment above nn_chain/3.
    pairwise =
      pairwise
      |> Nx.put_slice([0, i], Nx.reshape(updates, {n, 1}))
      |> Nx.put_slice([i, 0], Nx.reshape(updates, {1, n}))

    count = Nx.select(found, count + 1, count)

    {clades, count, pointers, pairwise, diss, sizes, cluster_sizes, alive}
  end

  # Dissimilarity update functions

  defnp average(dac, dbc, _dab, sa, sb, _sc),
    do: (sa * dac + sb * dbc) / (sa + sb)

  # defnp centroid(dac, dbc, dab, sa, sb, _sc),
  #   do: Nx.sqrt((sa * dac + sb * dbc) / (sa + sb) - sa * sb * dab / (sa + sb) ** 2)

  defnp complete(dac, dbc, _dab, _sa, _sb, _sc),
    do: Nx.max(dac, dbc)

  # defnp median(dac, dbc, dab, _sa, _sb, _sc),
  #   do: Nx.sqrt(dac / 2 + dbc / 2 - dab / 4)

  defnp single(dac, dbc, _dab, _sa, _sb, _sc),
    do: Nx.min(dac, dbc)

  defnp ward(dac, dbc, dab, sa, sb, sk),
    do: Nx.sqrt(((sa + sk) * dac ** 2 + (sb + sk) * dbc ** 2 - sk * dab ** 2) / (sa + sb + sk))

  defnp weighted(dac, dbc, _dab, _sa, _sb, _sc),
    do: (dac + dbc) / 2

  # Cluster label functions

  @label_opts_schema [
    cluster_by: [
      type: :non_empty_keyword_list,
      required: true,
      keys: [
        height: [
          type: :float,
          doc: "Height of the dendrogram to use as the split point for clusters."
        ],
        num_clusters: [
          type: :pos_integer,
          doc: "Number of clusters to form."
        ]
      ],
      doc: """
      How to select which clades from the dendrogram should form the final clusters.
      Must provide either a height or a number of clusters.
      """
    ]
  ]
  @doc """
  Cluster a `Scholar.Cluster.Hierarchical` struct into a map of cluster labels to member indices.

  ## Options

  #{NimbleOptions.docs(@label_opts_schema)}

  ## Return values

  Returns a map where the keys are integers from `0..(k - 1)` where `k` is the number of clusters.
  Each value is a cluster represented by a list of member indices.
  E.g. if the result map was `%{0 => [0, 1], 1 => [2]}`, then elements `[0, 1]` of the data would
  be in cluster `0` and the singleton element `[2]` would be in cluster `1`.

  Cluster labels are arbitrary, but deterministic.

  ## Examples

      iex> data = Nx.tensor([[5], [5], [5], [10], [10]])
      iex> model = Hierarchical.fit(data)
      iex> Hierarchical.labels_map(model, cluster_by: [num_clusters: 2])
      %{0 => [0, 1, 2], 1 => [3, 4]}
  """
  def labels_map(%__MODULE__{} = model, opts) do
    opts = NimbleOptions.validate!(opts, @label_opts_schema)

    raw_clusters =
      case opts[:cluster_by] do
        [height: height] ->
          cluster_by_height(model, height)

        [num_clusters: num_clusters] ->
          cond do
            num_clusters > model.num_points ->
              raise ArgumentError, "`num_clusters` may not exceed number of data points."

            num_clusters == model.num_points ->
              Nx.broadcast(0, {model.num_points})

            # The other cases are validated by NimbleOptions.
            true ->
              cluster_by_num_clusters(model, num_clusters)
          end

        _ ->
          raise ArgumentError, "Must pass exactly one of `:height` or `:num_clusters`"
      end

    # Give the clusters labels 0..(k - 1) and ensure those labels are deterministic by sorting by
    # the minimum element.
    raw_clusters
    |> Enum.sort_by(fn {_label, cluster} -> Enum.min(cluster) end)
    |> Enum.with_index()
    |> Enum.flat_map(fn {{_, v}, i} -> v |> Enum.sort() |> Enum.map(&{&1, i}) end)
    |> Enum.group_by(fn {_, v} -> v end, fn {k, _} -> k end)
  end

  @doc """
  Cluster a `Scholar.Cluster.Hierarchical` struct into a list of cluster labels.

  ## Options

  #{NimbleOptions.docs(@label_opts_schema)}

  ## Return values

  Returns a list of length `n` and values `0..(k - 1)` where `n` is the number of data points and
  `k` is the number of clusters formed.
  The `i`th element of the result list is the label of the `i`th data point's cluster.

  Cluster labels are arbitrary, but deterministic.

  ## Examples

      iex> data = Nx.tensor([[5], [5], [5], [10], [10]])
      iex> model = Hierarchical.fit(data)
      iex> Hierarchical.labels_list(model, cluster_by: [num_clusters: 2])
      [0, 0, 0, 1, 1]
  """
  def labels_list(%__MODULE__{} = model, opts) do
    model
    |> labels_map(opts)
    |> Enum.flat_map(fn {k, vs} -> Enum.map(vs, &{&1, k}) end)
    |> Enum.sort()
    |> Enum.map(fn {_, v} -> v end)
  end

  defp cluster_by_height(model, height_cutoff) do
    clusters = Map.new(0..(model.num_points - 1), &{&1, [&1]})

    Enum.zip(Nx.to_list(model.clades), Nx.to_list(model.dissimilarities))
    |> Enum.with_index(model.num_points)
    |> Enum.reduce_while(clusters, fn {{[a, b], height}, c}, clusters ->
      if height >= height_cutoff do
        {:halt, clusters}
      else
        {:cont, merge_clusters(clusters, a, b, c)}
      end
    end)
  end

  defp cluster_by_num_clusters(model, num_clusters) do
    clusters = Map.new(0..(model.num_points - 1), &{&1, [&1]})

    Nx.to_list(model.clades)
    |> Enum.with_index(model.num_points)
    |> Enum.reduce_while(clusters, fn {[a, b], c}, clusters ->
      if c + num_clusters == 2 * model.num_points do
        {:halt, clusters}
      else
        {:cont, merge_clusters(clusters, a, b, c)}
      end
    end)
  end

  defp merge_clusters(clusters, a, b, c) do
    clusters
    |> Map.put(c, clusters[a] ++ clusters[b])
    |> Map.drop([a, b])
  end
end
