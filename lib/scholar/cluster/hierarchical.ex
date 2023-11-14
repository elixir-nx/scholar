defmodule Scholar.Cluster.Hierarchical do
  @moduledoc """
  https://arxiv.org/abs/1109.2378
  """
  import Nx.Defn

  alias Scholar.Cluster.Hierarchical.CondensedMatrix

  @dissimilarity_types [
    :euclidean,
    :precomputed
  ]

  @linkage_types [
    :average,
    :centroid,
    :complete,
    :median,
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
        :euclidean -> &pairwise_euclidean/1
        :precomputed -> &Function.identity/1
      end

    update_fun =
      case linkage do
        :average -> &average/6
        :centroid -> &centroid/6
        :complete -> &complete/6
        :median -> &median/6
        :single -> &single/6
        :ward -> &ward/6
        :weighted -> &weighted/6
      end

    cluster_fun =
      case linkage do
        # TODO: change to `mst`
        # :single -> &primitive/2
        :single -> &nearest_neighbor_chain/2
        # TODO: change to `nn_chain`
        l when l in [:average, :complete, :ward, :weighted] -> &primitive/2
        # TODO: change to `generic`
        l when l in [:centroid, :median] -> &primitive/2
      end

    dendrogram =
      data
      |> dissimilarity_fun.()
      |> cluster_fun.(update_fun)

    labels =
      if group_by do
        group_fun =
          case group_by do
            [height: height] -> &group_by_height(&1, height)
            [num_clusters: num_clusters] -> &group_by_num_clusters(&1, num_clusters)
          end

        dendrogram |> group_fun.() |> groups_to_labels()
      else
        nil
      end

    %{
      dendrogram: dendrogram,
      labels: labels
    }
  end

  # Cluster functions

  @doc """
  The "primitive" clustering procedure.

  This is figure 1 of the paper (p. 6). It serves as the baseline for the other algorithms.

  It has a best case complexity of O(n^3) and shouldn't be used in practice.
  """
  def primitive(%Nx.Tensor{} = pd, update_fun) do
    {n_row, n_row} = Nx.shape(pd)

    # Cluster labels.
    labels = MapSet.new(0..(n_row - 1))

    # Cluster sizes.
    size = Map.new(0..(n_row - 1), &{&1, 1})

    # Dissimilarities between each pair of data points.
    # Map of %{[i, j] => dissimiliarity} where i, j are the indices of the two data.
    ci = n_row |> CondensedMatrix.pairwise_indices() |> Nx.to_list()
    cd = pd |> CondensedMatrix.condense_pairwise() |> Nx.to_list()
    d = [ci, cd] |> Enum.zip() |> Map.new()

    {output, _size, _labels, _d, _n} =
      for _ <- 0..(n_row - 2), reduce: {[], size, labels, d, n_row - 1} do
        {output, size, labels, d, n_old} ->
          # Create a new cluster label n.
          n = n_old + 1

          # Find the closest pair of old clusters.
          # This pair will be merged to form cluster n.
          {[a, b], dab} = Enum.min_by(d, fn {i, d} -> {d, i} end)

          # Add new cluster to output.
          output = [{n, [a, b], dab} | output]

          # Remove old cluster labels.
          labels = Enum.reduce([a, b], labels, &MapSet.delete(&2, &1))

          # Update dissimilarities.
          d =
            for c <- labels, reduce: d do
              d ->
                [ac, bc] = [[a, c], [b, c]] |> Enum.map(&Enum.sort(&1, :desc))
                update = update_fun.(d[ac], d[bc], d[[a, b]], size[a], size[b], size[c])

                d
                # [n, c] is ordered correctly because n > c always.
                |> Map.put([n, c], update)
                |> Map.drop([ac, bc])
            end

          d = Map.delete(d, [a, b])

          # Update sizes.
          size = Map.put(size, n, size[a] + size[b])

          # Add n to the labels.
          labels = MapSet.put(labels, n)

          {output, size, labels, d, n}
      end

    Enum.reverse(output)
  end

  def mst(%Nx.Tensor{} = _pd, _update_fun), do: raise("not yet implemented")
  def nn_chain(%Nx.Tensor{} = _pd, _update_fun), do: raise("not yet implemented")
  def generic(%Nx.Tensor{} = _pd, _update_fun), do: raise("not yet implemented")

  # Dissimilarity functions

  def pairwise_euclidean(%Nx.Tensor{} = x) do
    x |> pairwise_euclidean_sq() |> Nx.sqrt()
  end

  defn pairwise_euclidean_sq(%Nx.Tensor{} = x) do
    sq = Nx.sum(x ** 2, axes: [1], keep_axes: true)
    sq + Nx.transpose(sq) - 2 * Nx.dot(x, [1], x, [1])
  end

  # Dissimilarity update functions

  def average(dac, dbc, _dab, na, nb, _nc),
    do: (na * dac + nb * dbc) / (na + nb)

  def centroid(dac, dbc, dab, na, nb, _nc),
    do: Nx.sqrt((na * dac + nb * dbc) / (na + nb) - na * nb * dab / (na + nb) ** 2)

  def complete(dac, dbc, _dab, _na, _nb, _nc),
    do: max(dac, dbc)

  def median(dac, dbc, dab, _na, _nb, _nc),
    do: Nx.sqrt(dac / 2 + dbc / 2 - dab / 4)

  def single(dac, dbc, _dab, _na, _nb, _nc),
    do: min(dac, dbc)

  def ward(dac, dbc, dab, na, nb, nc),
    do: Nx.sqrt(((na + nc) * dac + (nb + nc) * dbc - nc * dab) / (na + nb + nc))

  def weighted(dac, dbc, _dab, _na, _nb, _nc),
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

  defn nearest_neighbor_chain(pairwise, update_fun) do
    {clusters, distances} = nearest_neighbor_core(pairwise, update_fun)
    permutation = Nx.argsort(distances)
    clusters = Nx.take(clusters, permutation)
    distances = Nx.take(distances, permutation)
    {relabel(clusters), distances}
  end

  defn nearest_neighbor_core(pairwise, _update_fun) do
    {num_obs, _} = Nx.shape(pairwise)
    clusters = Nx.broadcast(Nx.s64(0), {num_obs - 1, 2})
    distances = Nx.broadcast(Nx.Constants.nan(Nx.type(pairwise)), {num_obs - 1})
    pairwise = :infinity |> Nx.broadcast({num_obs}) |> Nx.make_diagonal() |> Nx.add(pairwise)
    labels = Nx.iota({num_obs})

    {clusters, distances, _, _, _} =
      while {clusters, distances, pairwise, labels, len = 3}, n <- 0..(num_obs - 2) do
        {{a0, b0}, len} = if len >= 3, do: {find_pair(labels), 2}, else: {{-1, -1}, len - 3}

        # The key idea is that we recursively ask for the argmin of a row.
        # When we find an a-b-a cycle in the argmin chain, it means a and b are a pair.
        {a, b, _, len, _} =
          while {a = a0, b = b0, c = -1, len, pairwise}, not (len >= 3 and a == c) do
            {Nx.argmin(pairwise[a]), a, b, len + 1, pairwise}
          end

        clusters = Nx.put_slice(clusters, [n, 0], [a, b] |> Nx.stack() |> Nx.new_axis(0))
        distances = Nx.indexed_put(distances, Nx.stack([n]), pairwise[[a, b]])
        {pairwise, labels} = merge(pairwise, labels, a, b)

        {clusters, distances, pairwise, labels, len}
      end

    {clusters, distances}
  end

  # @infinite_index represents an removed label.
  # Can't use :infinity because we need integers.
  @infinite_index Nx.Constants.max_finite(:u32)
  defn find_pair(labels) do
    # Choose the earliest two not-yet-removed labels.
    a = Nx.argmin(labels)
    b = labels |> Nx.indexed_put(Nx.new_axis(a, 0), @infinite_index) |> Nx.argmin()
    {a, b}
  end

  defn merge(pairwise, labels, label_1, label_2) do
    {num_obs, _} = Nx.shape(pairwise)
    a = Nx.min(label_1, label_2)
    b = Nx.max(label_1, label_2)

    {pairwise, _, _, _} =
      while {pairwise, a, b, inf = @infinite_index}, c <- labels do
        if c == a or c == b or c == inf do
          {pairwise, a, b, inf}
        else
          # TODO: generalize updates (hardcoded for single linkage)
          update = Nx.min(pairwise[[a, c]], pairwise[[b, c]])

          # TODO: tensor-ify?
          pairwise =
            pairwise
            |> Nx.indexed_put(Nx.stack([a, c]), update)
            |> Nx.indexed_put(Nx.stack([c, a]), update)
            |> Nx.indexed_put(Nx.stack([b, c]), update)
            |> Nx.indexed_put(Nx.stack([c, b]), update)

          {pairwise, a, b, inf}
        end
      end

    # Drop the greater of a and b (by convention, a < b)
    pairwise =
      pairwise
      |> Nx.put_slice([b, 0], Nx.broadcast(:infinity, {1, num_obs}))
      |> Nx.put_slice([0, b], Nx.broadcast(:infinity, {num_obs, 1}))

    labels =
      labels
      |> Nx.indexed_put(Nx.new_axis(a, 0), @infinite_index)
      |> Nx.indexed_put(Nx.new_axis(b, 0), @infinite_index)

    {pairwise, labels}
  end

  defn relabel(clusters) do
    {num_obs, 2} = Nx.shape(clusters)
    num_obs = num_obs + 1
    parent = Nx.iota({num_obs})
    labels = Nx.broadcast(-1, {num_obs - 1, 3})

    {labels, _, _} =
      while {labels, parent, clusters}, i <- 0..(num_obs - 2) do
        a = clusters[[i, 0]]
        b = clusters[[i, 1]]
        x = parent[a]
        y = parent[b]
        labels = Nx.put_slice(labels, [i, 0], [i + num_obs, x, y] |> Nx.stack() |> Nx.new_axis(0))
        parent = Nx.indexed_put(parent, Nx.stack([a]), i + num_obs)
        parent = Nx.indexed_put(parent, Nx.stack([b]), i + num_obs)
        {labels, parent, clusters}
      end

    labels
  end
end
