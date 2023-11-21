defmodule Scholar.Cluster.Hierarchical do
  @moduledoc """
  Performs [hierarchical, agglomerative clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering#Agglomerative_clustering_example)
  on a dataset.
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
      Used to build an `{n, n}` shaped, symmetric `Nx.Tensor`.
      """
    ],
    linkage: [
      type: {:in, @linkage_types},
      default: :single,
      doc:
        "Linkage function: how to compute the dissimilarity between a newly formed clade and the others."
    ]
  ]
  @doc """
  Use hierarchical clustering to form the initial clades to be clustered using `cluster/2`.

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
  """
  deftransform fit(data, opts \\ []) do
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

    pairwise = dissimilarity_fun.(data)

    n =
      case Nx.shape(pairwise) do
        {n, n} -> n
        _ -> raise ArgumentError, "pairwise must be a symmetric matrix"
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

    {clades, _count, _pointers, _pairwise, diss, sizes} =
      while {clades, count = 0, pointers, pairwise, diss, sizes}, count < n - 1 do
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

        {clades, count, pointers, pairwise, diss, sizes}
      end

    sizes = sizes[n..(2 * n - 2)]
    perm = Nx.argsort(diss)
    {clades[perm], diss[perm], sizes[perm]}
  end

  defnp merge_clades(clades, count, pointers, pairwise, diss, sizes, links, n, update_fun) do
    {clades, count, pointers, pairwise, diss, sizes, _links} =
      while {clades, count, pointers, pairwise, diss, sizes, links},
            i <- 0..(Nx.size(links) - 1) do
        # i < j because of how links is formed.
        # i will become the new clade index and we "infinity-out" j.
        j = links[i]

        if j == n do
          {clades, count, pointers, pairwise, diss, sizes, links}
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
          {pairwise, _, _, _, _, _} =
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

  # Cluster functions

  @cluster_opts_schema [
    by: [
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
  Cluster the dataset using the clades formed by `fit/2`.

  ## Options

  #{NimbleOptions.docs(@cluster_opts_schema)}

  ## Return values

  Returns a `Nx.Tensor` with shape `{n}` and values `0..(n - 1)` where `n == model.num_points`.
  The `i`th element of the result tensor is the index of that data point's cluster.
  """
  deftransform cluster(%__MODULE__{} = model, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @cluster_opts_schema)

    clusters =
      case opts[:by] do
        [height: height] ->
          cluster_by_height(model, height)

        [num_clusters: num_clusters] ->
          cond do
            num_clusters > model.num_points ->
              raise ArgumentError, "`num_clusters` may not exceed number of data points"

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
