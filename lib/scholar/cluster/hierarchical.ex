defmodule Scholar.Cluster.Hierarchical do
  @moduledoc """
  https://arxiv.org/abs/1109.2378
  """
  import Nx.Defn

  defstruct [:clusters, :dissimilarities, :labels, :sizes]

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

  @opts_schema [
    dissimilarity: [
      type: {:in, @dissimilarity_types},
      default: :euclidean,
      doc:
        "Pairwise dissimilarity function: computes the 'dissimilarity' between each pair of data points."
    ],
    group_by: [
      type: :non_empty_keyword_list,
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
      How to group the dendrogram into clusters.
      Must provide either a height or a number of clusters.
      """
    ],
    linkage: [
      type: {:in, @linkage_types},
      default: :single,
      doc:
        "Linkage function: how to compute the dissimilarity between a newly formed cluster and the others."
    ]
  ]
  deftransform fit(data, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    dissimilarity = opts[:dissimilarity]
    group_by = opts[:group_by]
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

    cluster_fun =
      case linkage do
        # TODO: :centroid, :median
        l when l in [:average, :complete, :single, :ward, :weighted] ->
          &parallel_nearest_neighbor/2
      end

    pairwise = dissimilarity_fun.(data)

    if not match?({n, n}, Nx.shape(pairwise)) do
      raise ArgumentError, "pairwise must be a symmetric matrix"
    end

    {clusters, diss, sizes} = cluster_fun.(pairwise, update_fun)

    labels =
      if group_by do
        # groups =
        #   case group_by do
        #     [height: height] ->
        #       group_by_height(clusters, diss, height)
        #
        #     [num_clusters: num_clusters] ->
        #       group_by_num_clusters(clusters, diss, num_clusters)
        #   end

        # groups_to_labels(groups)
        nil
      else
        nil
      end

    %__MODULE__{
      clusters: clusters,
      dissimilarities: diss,
      labels: labels,
      sizes: sizes
    }
  end

  # Cluster functions

  defnp parallel_nearest_neighbor(pairwise, update_fun) do
    {n, _} = Nx.shape(pairwise)
    pairwise = Nx.broadcast(:infinity, {n}) |> Nx.make_diagonal() |> Nx.add(pairwise)
    clusters = Nx.broadcast(-1, {n - 1, 2})
    sizes = Nx.broadcast(1, {2 * n - 1})
    pointers = Nx.broadcast(-1, {2 * n - 2})
    diss = Nx.tensor(:infinity, type: Nx.type(pairwise)) |> Nx.broadcast({n - 1})

    {clusters, _count, _pointers, _pairwise, diss, sizes} =
      while {clusters, count = 0, pointers, pairwise, diss, sizes}, count < n - 1 do
        # Indexes of who I am nearest to
        nearest = Nx.argmin(pairwise, axis: 1, type: :u32)

        # Take who I am nearest to is nearest to
        nearest_of_nearest = Nx.take(nearest, nearest)

        # If the entry is pointing back at me, then we are a cluster
        clusters_selector = Nx.equal(nearest_of_nearest, Nx.iota({n}))

        # Now let's get the links that form clusters.
        # They are bidirectional but let's keep only one side.
        links = Nx.select(clusters_selector and nearest > nearest_of_nearest, nearest, n)

        {clusters, count, pointers, pairwise, diss, sizes} =
          merge_clusters(clusters, count, pointers, pairwise, diss, sizes, links, n, update_fun)

        {clusters, count, pointers, pairwise, diss, sizes}
      end

    {clusters, diss, sizes[n..(2 * n - 2)]}
  end

  defnp merge_clusters(clusters, count, pointers, pairwise, diss, sizes, links, n, update_fun) do
    {clusters, count, pointers, pairwise, diss, sizes, _links} =
      while {clusters, count, pointers, pairwise, diss, sizes, links},
            i <- 0..(Nx.size(links) - 1) do
        # i < j because of how links is formed.
        # i will become the new cluster index and we "infinity-out" j.
        j = links[i]

        if j == n do
          {clusters, count, pointers, pairwise, diss, sizes, links}
        else
          # Clusters a and b (i and j of pairwise) are being merged into c.
          a = find_cluster(pointers, i)
          b = find_cluster(pointers, j)
          c = count + n

          # Update clusters
          new_cluster = Nx.stack([a, b]) |> Nx.sort() |> Nx.new_axis(0)
          clusters = Nx.put_slice(clusters, [count, 0], new_cluster)

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

          {clusters, count + 1, pointers, pairwise, diss, sizes, links}
        end
      end

    {clusters, count, pointers, pairwise, diss, sizes}
  end

  defnp find_cluster(pointers, i) do
    {i, _, _} =
      while {_current = i, next = pointers[i], pointers}, next != -1 do
        {next, pointers[next], pointers}
      end

    i
  end

  # Dissimilarity functions

  defn pairwise_euclidean(%Nx.Tensor{} = x) do
    x |> pairwise_euclidean_sq() |> Nx.sqrt()
  end

  defn pairwise_euclidean_sq(%Nx.Tensor{} = x) do
    sq = Nx.sum(x ** 2, axes: [1], keep_axes: true)
    sq + Nx.transpose(sq) - 2 * Nx.dot(x, [1], x, [1])
  end

  # Dissimilarity update functions

  defn average(dac, dbc, _dab, sa, sb, _sc),
    do: (sa * dac + sb * dbc) / (sa + sb)

  # defn centroid(dac, dbc, dab, sa, sb, _sc),
  #   do: Nx.sqrt((sa * dac + sb * dbc) / (sa + sb) - sa * sb * dab / (sa + sb) ** 2)

  defn complete(dac, dbc, _dab, _sa, _sb, _sc),
    do: Nx.max(dac, dbc)

  # defn median(dac, dbc, dab, _sa, _sb, _sc),
  #   do: Nx.sqrt(dac / 2 + dbc / 2 - dab / 4)

  defn single(dac, dbc, _dab, _sa, _sb, _sc),
    do: Nx.min(dac, dbc)

  defn ward(dac, dbc, dab, sa, sb, sc),
    do: Nx.sqrt(((sa + sc) * dac + (sb + sc) * dbc - sc * dab) / (sa + sb + sc))

  defn weighted(dac, dbc, _dab, _sa, _sb, _sc),
    do: (dac + dbc) / 2

  # Grouping functions

  defp group_by_height([{count, _, _} | _] = dendrogram, height_cutoff) do
    clusters = Map.new(0..(count - 1), &{&1, [&1]})

    Enum.reduce_while(dendrogram, clusters, fn {c, [a, b], height}, clusters ->
      if height >= height_cutoff do
        {:halt, clusters}
      else
        clusters =
          clusters
          |> Map.put(c, clusters[a] ++ clusters[b])
          |> Map.drop([a, b])

        {:cont, clusters}
      end
    end)
  end

  defp group_by_num_clusters([{count, _, _} | _] = dendrogram, num_clusters) do
    clusters = Map.new(0..(count - 1), &{&1, [&1]})

    Enum.reduce_while(dendrogram, {clusters, count}, fn {c, [a, b], _}, {clusters, count} ->
      if count == num_clusters do
        {:halt, clusters}
      else
        clusters =
          clusters
          |> Map.put(c, clusters[a] ++ clusters[b])
          |> Map.drop([a, b])

        {:cont, {clusters, count - 1}}
      end
    end)
  end

  defp groups_to_labels(groups) do
    groups
    |> Enum.sort_by(fn {_label, group} -> Enum.min(group) end)
    |> Enum.with_index()
    |> Enum.flat_map(fn {{_, v}, i} -> v |> Enum.sort() |> Enum.map(&{&1, i}) end)
    |> Enum.sort()
    |> Enum.map(fn {_, label} -> label end)
    |> Nx.tensor()
  end
end
