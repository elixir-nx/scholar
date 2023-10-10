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
    linkage: [
      type: {:in, @linkage_types},
      default: :single,
      doc:
        "Linkage function: how to compute the dissimilarity between a newly formed cluster and the others."
    ]
  ]
  def fit(data, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    dissimilarity = opts[:dissimilarity]
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
        :single -> &primitive/2
        # TODO: change to `nn_chain`
        l when l in [:average, :complete, :ward, :weighted] -> &primitive/2
        # TODO: change to `generic`
        l when l in [:centroid, :median] -> &primitive/2
      end

    data
    |> dissimilarity_fun.()
    |> cluster_fun.(update_fun)
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

  def average(dac, dbc, _dab, na, nb, _nc), do: (na * dac + nb * dbc) / (na + nb)
  def centroid(_dac, _dbc, _dab, _na, _nb, _nc), do: raise("not yet implemented")
  def complete(dac, dbc, _dab, _na, _nb, _nc), do: max(dac, dbc)
  def median(_dac, _dbc, _dab, _na, _nb, _nc), do: raise("not yet implemented")
  def single(dac, dbc, _dab, _na, _nb, _nc), do: min(dac, dbc)
  def ward(_dac, _dbc, _dab, _na, _nb, _nc), do: raise("not yet implemented")
  def weighted(_dac, _dbc, _dab, _na, _nb, _nc), do: raise("not yet implemented")
end
