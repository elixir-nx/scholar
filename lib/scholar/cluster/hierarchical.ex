defmodule Scholar.Cluster.Hierarchical do
  @moduledoc """
  Performs [hierarchical, agglomerative clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering#Agglomerative_clustering_example)
  on a dataset.

  Hierarchical clustering is good for when the number of clusters is not known ahead of time.
  It also allows for the creation of a [dendrogram plot](https://en.wikipedia.org/wiki/Dendrogram)
  (regardless of the dimensionality of the dataset) which can be used to select the number of
  clusters in a post-processing step.

  <!-- (This isn't ready yet.)

  ## Examples

  See the [_Hierarchical Clustering_](./doc/notebooks/hierarchical_clustering.livemd) Livebook for
  a worked example with a dendrogram plot.

  -->

  ## Limitations

  Due to the requirements of the current implementation, only these options are supported:

    * `dissimilarity: :euclidean`
    * `linkage: :average | :complete | :single | :ward | :weighted`

  Our current algorithm is $O(\\frac{n^2}{p})$ where $n$ is the number of data points
  and $p$ is the number of processors.
  This is better than the generic algorithm which is $O(n^3)$ and even other specialized algorithms
  like SLINK (see [here](https://en.wikipedia.org/wiki/Single-linkage_clustering)) which are
  $O(n^2)$.
  However, it requires certain theoretical properties of the dissimilarities and linkages.
  As such, we've restricted the options to only those combinations with the correct properties.

  In the future, we plan to add additional algorithms which will be slower but won't have the
  same restrictions.
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
  Use hierarchical clustering to form the initial model to be clustered using `fit_predict/2`.

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
        :euclidean -> &pairwise_euclidean/1
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
          &parallel_nearest_neighbor/2
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

    pairwise = dissimilarity_fun.(data)

    if Nx.shape(pairwise) != {n, n} do
      raise ArgumentError, "Pairwise must be a symmetric matrix."
    end

    {clades, diss, sizes} = dendrogram_fun.(pairwise, update_fun)

    %__MODULE__{
      clades: clades,
      dissimilarities: diss,
      num_points: n,
      sizes: sizes
    }
  end

  # Clade functions

  defnp parallel_nearest_neighbor(pairwise, update_fun) do
    {n, _} = Nx.shape(pairwise)
    pairwise = Nx.broadcast(:infinity, {n}) |> Nx.make_diagonal() |> Nx.add(pairwise)
    clades = Nx.broadcast(-1, {n - 1, 2})
    sizes = Nx.broadcast(1, {2 * n - 1})
    pointers = Nx.broadcast(-1, {2 * n - 2})
    diss = Nx.tensor(:infinity, type: Nx.type(pairwise)) |> Nx.broadcast({n - 1})

    {clades, _} =
      while {clades, {count = 0, pointers, pairwise, diss, sizes}}, count < n - 1 do
        # Indexes of who I am nearest to
        nearest = Nx.argmin(pairwise, axis: 1)

        # Take who I am nearest to is nearest to
        nearest_of_nearest = Nx.take(nearest, nearest)

        # If the entry is pointing back at me, then we are a clade
        clades_selector = Nx.equal(nearest_of_nearest, Nx.iota({n}))

        # Now let's get the links that form clades.
        # They are bidirectional but let's keep only one side.
        links = Nx.select(clades_selector and nearest > nearest_of_nearest, nearest, n)

        {clades, count, pointers, pairwise, diss, sizes} =
          merge_clades(clades, count, pointers, pairwise, diss, sizes, links, n, update_fun)

        {clades, {count, pointers, pairwise, diss, sizes}}
      end

    sizes = sizes[n..(2 * n - 2)]
    perm = Nx.argsort(diss)
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
          a = find_clade(pointers, i)
          b = find_clade(pointers, j)
          c = count + n

          # Update clades
          new_clade = Nx.stack([a, b]) |> Nx.sort() |> Nx.new_axis(0)
          clades = Nx.put_slice(clades, [count, 0], new_clade)

          # Update sizes
          sc = sizes[i] + sizes[j]
          sizes = Nx.indexed_put(sizes, Nx.stack([i, c]) |> Nx.new_axis(-1), Nx.stack([sc, sc]))

          # Update dissimilarities
          diss = Nx.indexed_put(diss, Nx.stack([count]), pairwise[i][j])

          # Update pairwise
          {pairwise, _} =
            while {pairwise, x = i, y = j, sa = sizes[i], sb = sizes[j], sc = sc},
                  z <- 0..(n - 1) do
              if z == x or z == y or Nx.is_infinity(pairwise[[0, z]]) do
                {pairwise, x, y, sa, sb, sc}
              else
                dac = pairwise[[x, z]]
                dbc = pairwise[[y, z]]
                dab = pairwise[[x, y]]
                update = update_fun.(dac, dbc, dab, sa, sb, sc)

                # TODO: do this in a single call?
                pairwise =
                  pairwise
                  |> Nx.indexed_put(Nx.stack([x, z]), update)
                  |> Nx.indexed_put(Nx.stack([z, x]), update)
                  |> Nx.indexed_put(Nx.stack([y, z]), update)
                  |> Nx.indexed_put(Nx.stack([z, y]), update)

                {pairwise, x, y, sa, sb, sc}
              end
            end

          infinities = Nx.take_diagonal(pairwise)

          pairwise =
            pairwise
            |> Nx.put_slice([j, 0], Nx.reshape(infinities, {1, n}))
            |> Nx.put_slice([0, j], Nx.reshape(infinities, {n, 1}))

          # Update pointers
          indices = [i, j] |> Nx.stack() |> Nx.new_axis(-1)
          pointers = Nx.indexed_put(pointers, indices, Nx.stack([c, c]))

          {clades, count + 1, pointers, pairwise, diss, sizes, links}
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

  # Dissimilarity functions

  defnp pairwise_euclidean(%Nx.Tensor{} = x) do
    x |> pairwise_euclidean_sq() |> Nx.sqrt()
  end

  defnp pairwise_euclidean_sq(%Nx.Tensor{} = x) do
    sq = Nx.sum(x ** 2, axes: [1], keep_axes: true)
    sq + Nx.transpose(sq) - 2 * Nx.dot(x, [1], x, [1])
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

  defnp ward(dac, dbc, dab, sa, sb, sc),
    do: Nx.sqrt(((sa + sc) * dac + (sb + sc) * dbc - sc * dab) / (sa + sb + sc))

  defnp weighted(dac, dbc, _dab, _sa, _sb, _sc),
    do: (dac + dbc) / 2

  # Predict functions

  @predict_opts_schema [
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
  Cluster the dataset using hierarchical clustering or cluster the model built by `fit/2`.

  ## Options

  If the input is a `Scholar.Cluster.Hierarchical` struct, only these options are required:

  #{NimbleOptions.docs(@predict_opts_schema)}

  If the input is a `Nx.Tensor`, you must also pass the options required by `fit/2`.

  ## Return values

  Returns a `Nx.Tensor` with shape `{model.num_points}` and values `0..(k - 1)` where `k` is the
  number of clusters formed.
  The `i`th element of the result tensor is the label of the `i`th data point's cluster.
  (Cluster labels are arbitrary, but deterministic.)

  ## Examples

      iex> data = Nx.tensor([[2], [7], [9], [0], [3]])
      iex> model = Hierarchical.fit(data)
      iex> Hierarchical.fit_predict(model, cluster_by: [num_clusters: 3])
      Nx.tensor([0, 1, 1, 2, 0])
  """
  deftransform fit_predict(data_or_model, opts \\ [])

  deftransform fit_predict(%Nx.Tensor{} = data, opts) do
    {fit_opts, predict_opts} = Enum.split_with(opts, &(&1 in @fit_opts_schema))

    data
    |> fit(fit_opts)
    |> fit_predict(predict_opts)
  end

  deftransform fit_predict(%__MODULE__{} = model, opts) do
    opts = NimbleOptions.validate!(opts, @predict_opts_schema)

    clusters =
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
      end

    clusters_to_labels(clusters)
  end

  deftransformp cluster_by_height(model, height_cutoff) do
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

  deftransformp cluster_by_num_clusters(model, num_clusters) do
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

  deftransformp merge_clusters(clusters, a, b, c) do
    clusters
    |> Map.put(c, clusters[a] ++ clusters[b])
    |> Map.drop([a, b])
  end

  deftransformp clusters_to_labels(clusters) do
    clusters
    |> Enum.sort_by(fn {_label, cluster} -> Enum.min(cluster) end)
    |> Enum.with_index()
    |> Enum.flat_map(fn {{_, v}, i} -> v |> Enum.sort() |> Enum.map(&{&1, i}) end)
    |> Enum.sort()
    |> Enum.map(fn {_, label} -> label end)
    |> Nx.tensor()
  end
end
