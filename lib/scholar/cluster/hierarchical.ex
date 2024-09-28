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

    * `dissimilarity: :euclidean`
    * `linkage: :average | :complete | :single | :weighted`

  Our current algorithm is $O(\\frac{n^2}{p} \\cdot \\log(n))$ where $n$ is the number of data points
  and $p$ is the number of processors.
  This is better than the generic algorithm which is $O(n^3)$.
  It is also parallel, which means that runtime decreases in direct proportion to the number of
  processors.

  However, the implementation requires certain theoretical properties of the dissimilarities and
  linkages.
  As such, we've restricted the options to only those combinations with the correct properties.

  In the future, we plan to add additional algorithms which won't have the same restrictions.
  """
  import Nx.Defn

  defstruct [:clades, :dissimilarities, :num_points, :sizes]

  @dissimilarity_types [
    :euclidean
    # :precomputed
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

  ## Examples

      iex> data = Nx.tensor([[2], [7], [9], [0], [3]])
      iex> Hierarchical.fit(data)
      %Scholar.Cluster.Hierarchical{
        clades: Nx.tensor([[0, 4], [1, 2], [3, 5], [6, 7]]),
        dissimilarities: Nx.tensor([1.0, 2.0, 2.0, 4.0]),
        num_points: 5,
        sizes: Nx.tensor([2, 2, 3, 5])
      }
  """
  deftransform fit(%Nx.Tensor{} = data, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @fit_opts_schema)
    dissimilarity = opts[:dissimilarity]
    linkage = opts[:linkage]

    dissimilarity_fun =
      case dissimilarity do
        # :precomputed -> &Function.identity/1
        :euclidean -> &Scholar.Metrics.Distance.pairwise_euclidean/1
      end

    update_fun =
      case linkage do
        :average -> &average/6
        # :centroid -> &centroid/6
        :complete -> &complete/6
        # :median -> &median/6
        :single -> &single/6
        # :ward -> &ward/6
        :weighted -> &weighted/6
      end

    dendrogram_fun =
      case linkage do
        # TODO: :centroid, :median, :ward
        l when l in [:average, :complete, :single, :weighted] ->
          &parallel_nearest_neighbor/3
      end

    n =
      case Nx.shape(data) do
        {n, _num_features} ->
          n

        other ->
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

  defnp parallel_nearest_neighbor(data, dissimilarity_fun, update_fun) do
    pairwise = dissimilarity_fun.(data)
    {n, _} = Nx.shape(pairwise)
    pairwise = Nx.broadcast(:infinity, {n}) |> Nx.make_diagonal() |> Nx.add(pairwise)
    clades = Nx.broadcast(-1, {n - 1, 2})
    sizes = Nx.broadcast(1, {2 * n - 1})
    pointers = Nx.broadcast(-1, {2 * n - 2})
    diss = Nx.tensor(:infinity, type: Nx.type(pairwise)) |> Nx.broadcast({n - 1})

    {{clades, diss, sizes}, _} =
      while {{clades, diss, sizes}, {count = 0, pointers, pairwise}}, count < n - 1 do
        # Indexes of who I am nearest to
        nearest = Nx.argmin(pairwise, axis: 1)

        # Take who I am nearest to is nearest to
        nearest_of_nearest = Nx.take(nearest, nearest)

        # If the entry is pointing back at me, then we are a clade
        clades_selector = nearest_of_nearest == Nx.iota({n})

        # Now let's get the links that form clades.
        # They are bidirectional but let's keep only one side.
        links = Nx.select(clades_selector and nearest > nearest_of_nearest, nearest, n)

        {clades, count, pointers, pairwise, diss, sizes} =
          merge_clades(clades, count, pointers, pairwise, diss, sizes, links, n, update_fun)

        {{clades, diss, sizes}, {count, pointers, pairwise}}
      end

    sizes = sizes[n..(2 * n - 2)]
    perm = Nx.argsort(diss, stable: false, type: :u32)
    {clades[perm], diss[perm], sizes[perm]}
  end

  defnp merge_clades(clades, count, pointers, pairwise, diss, sizes, links, n, update_fun) do
    {{clades, count, pointers, pairwise, diss, sizes}, _} =
      while {{clades, count, pointers, pairwise, diss, sizes}, links},
            i <- 0..(Nx.size(links) - 1) do
        # i < j because of how links is formed.
        # i will become the new clade index and we "infinity-out" j.
        j = links[i]

        if j == n do
          {{clades, count, pointers, pairwise, diss, sizes}, links}
        else
          # Clades a and b (i and j of pairwise) are being merged into c.
          indices = [i, j] |> Nx.stack() |> Nx.new_axis(-1)
          a = find_clade(pointers, i)
          b = find_clade(pointers, j)
          c = count + n

          # Update clades
          new_clade = Nx.stack([a, b]) |> Nx.sort() |> Nx.new_axis(0)
          clades = Nx.put_slice(clades, [count, 0], new_clade)

          # Update sizes
          sa = sizes[i]
          sb = sizes[j]
          sc = sa + sb
          sizes = Nx.indexed_put(sizes, Nx.stack([i, c]) |> Nx.new_axis(-1), Nx.stack([sc, sc]))

          # Update dissimilarities
          diss = Nx.indexed_put(diss, Nx.stack([count]), pairwise[i][j])

          # Update pointers
          pointers = Nx.indexed_put(pointers, indices, Nx.stack([c, c]))

          # Update pairwise
          updates =
            update_fun.(pairwise[i], pairwise[j], pairwise[i][j], sa, sb, sc)
            |> Nx.indexed_put(indices, Nx.broadcast(:infinity, {2}))

          pairwise =
            pairwise
            |> Nx.put_slice([i, 0], Nx.reshape(updates, {1, n}))
            |> Nx.put_slice([0, i], Nx.reshape(updates, {n, 1}))
            |> Nx.put_slice([j, 0], Nx.broadcast(:infinity, {1, n}))
            |> Nx.put_slice([0, j], Nx.broadcast(:infinity, {n, 1}))

          {{clades, count + 1, pointers, pairwise, diss, sizes}, links}
        end
      end

    {clades, count, pointers, pairwise, diss, sizes}
  end

  defnp find_clade(pointers, i) do
    {i, _, _} =
      while {_current = i, next = pointers[i], pointers}, next != -1 do
        {next, pointers[next], pointers}
      end

    i
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

  # defnp ward(dac, dbc, dab, sa, sb, sc),
  #   do: Nx.sqrt(((sa + sc) * dac + (sb + sc) * dbc - sc * dab) / (sa + sb + sc))

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
