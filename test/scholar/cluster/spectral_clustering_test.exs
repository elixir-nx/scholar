defmodule Scholar.Cluster.SpectralClusteringTest do
  use Scholar.Case, async: true

  alias Scholar.Cluster.SpectralClustering

  doctest SpectralClustering

  defp key, do: Nx.Random.key(42)

  # Compares two label assignments as partitions, i.e. up to a permutation of
  # the label ids (which are arbitrary for clustering algorithms).
  defp assert_same_partition(labels, reference) do
    assert canonical_partition(Nx.to_flat_list(labels)) == canonical_partition(reference)
  end

  defp canonical_partition(labels) do
    labels
    |> Enum.reduce({%{}, []}, fn label, {mapping, acc} ->
      mapping = Map.put_new(mapping, label, map_size(mapping))
      {mapping, [mapping[label] | acc]}
    end)
    |> elem(1)
    |> Enum.reverse()
  end

  describe "spectral embedding" do
    # A connected graph with distinct Laplacian eigenvalues, so the embedding
    # is unique (up to sign, fixed by the deterministic sign flip) and can be
    # compared against scikit-learn value by value. Reference values from
    # sklearn 1.6.1 (rbf affinity, normalized Laplacian, u = D^-1/2 v).
    test "matches sklearn on a connected graph (f64)" do
      x =
        Nx.tensor(
          [
            [0.0, 0.0],
            [0.3, 0.2],
            [0.2, 0.4],
            [0.4, 0.1],
            [1.5, 1.4],
            [1.7, 1.6],
            [1.6, 1.8],
            [1.8, 1.5]
          ],
          type: :f64
        )

      model = SpectralClustering.fit(x, num_clusters: 2, key: key())

      expected =
        Nx.tensor(
          [
            [0.21119082386324414, 0.22170302981050768],
            [0.21119082386324428, 0.21243951255638854],
            [0.21119082386324425, 0.20868001640905082],
            [0.21119082386324428, 0.21225159060755805],
            [0.2111908238632471, -0.19700947261064472],
            [0.21119082386324736, -0.21205548793328316],
            [0.2111908238632475, -0.21380727833466814],
            [0.2111908238632474, -0.21197409305730497]
          ],
          type: :f64
        )

      assert Nx.type(model.embedding) == {:f, 64}
      assert_all_close(model.embedding, expected, atol: 1.0e-5)
      assert_same_partition(model.labels, [1, 1, 1, 1, 0, 0, 0, 0])
    end

    test "matches sklearn with a precomputed affinity (f64)" do
      affinity =
        Nx.tensor(
          [
            [1.0, 0.9, 0.8, 0.1, 0.0],
            [0.9, 1.0, 0.85, 0.05, 0.1],
            [0.8, 0.85, 1.0, 0.1, 0.05],
            [0.1, 0.05, 0.1, 1.0, 0.95],
            [0.0, 0.1, 0.05, 0.95, 1.0]
          ],
          type: :f64
        )

      model =
        SpectralClustering.fit(affinity,
          num_clusters: 2,
          affinity: :precomputed,
          key: key()
        )

      expected =
        Nx.tensor(
          [
            [0.3580574370197165, -0.24303023799947535],
            [0.3580574370197158, -0.22694330499537213],
            [0.35805743701971593, -0.22451947994046756],
            [0.35805743701971704, 0.5360175324304737],
            [0.35805743701971726, 0.5723279389695742]
          ],
          type: :f64
        )

      assert_all_close(model.embedding, expected, atol: 1.0e-5)
      assert_same_partition(model.labels, [0, 0, 0, 1, 1])
    end
  end

  describe "labels" do
    # For well-separated blobs the graph is effectively disconnected and the
    # smallest eigenvalues are degenerate, so the embedding itself is only
    # unique up to a rotation. The partition, however, is stable and must
    # match scikit-learn's.
    test "two well-separated blobs match sklearn's partition" do
      x =
        Nx.tensor([
          [0.0, 0.0],
          [0.2, 0.1],
          [0.1, 0.3],
          [0.3, 0.2],
          [5.0, 5.0],
          [5.2, 5.1],
          [5.1, 5.3],
          [5.3, 5.2]
        ])

      model = SpectralClustering.fit(x, num_clusters: 2, key: key())
      assert_same_partition(model.labels, [1, 1, 1, 1, 0, 0, 0, 0])
      assert Nx.shape(model.embedding) == {8, 2}
    end

    test "three blobs match sklearn's partition" do
      x =
        Nx.tensor([
          [0.0, 0.0],
          [0.1, 0.2],
          [0.2, 0.1],
          [4.0, 4.0],
          [4.1, 4.2],
          [4.2, 4.1],
          [8.0, 0.0],
          [8.1, 0.2],
          [8.2, 0.1]
        ])

      model = SpectralClustering.fit(x, num_clusters: 3, key: key())
      assert_same_partition(model.labels, [1, 1, 1, 2, 2, 2, 0, 0, 0])
      assert Nx.shape(model.embedding) == {9, 3}
    end

    test "custom gamma matches sklearn's partition" do
      x =
        Nx.tensor([
          [0.0, 0.0],
          [0.2, 0.1],
          [0.1, 0.3],
          [0.3, 0.2],
          [5.0, 5.0],
          [5.2, 5.1],
          [5.1, 5.3],
          [5.3, 5.2]
        ])

      model = SpectralClustering.fit(x, num_clusters: 2, gamma: 0.5, key: key())
      assert_same_partition(model.labels, [1, 1, 1, 1, 0, 0, 0, 0])
    end

    test "disconnected graph: two separate cliques are split (precomputed)" do
      affinity =
        Nx.tensor(
          [
            [1.0, 0.9, 0.8, 0.0, 0.0, 0.0],
            [0.9, 1.0, 0.85, 0.0, 0.0, 0.0],
            [0.8, 0.85, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.7, 0.6],
            [0.0, 0.0, 0.0, 0.7, 1.0, 0.75],
            [0.0, 0.0, 0.0, 0.6, 0.75, 1.0]
          ],
          type: :f64
        )

      model =
        SpectralClustering.fit(affinity, num_clusters: 2, affinity: :precomputed, key: key())

      assert_same_partition(model.labels, [0, 0, 0, 1, 1, 1])
    end

    test "isolated vertex gets its own cluster, matching sklearn (precomputed)" do
      # Node 4 has no edges (all off-diagonal affinities are 0). Its Laplacian
      # diagonal is set to 1 (as scikit-learn's `_set_diag` does), so it does not
      # produce a spurious zero eigenvalue nor a NaN.
      affinity =
        Nx.tensor(
          [
            [1.0, 0.9, 0.0, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.85, 0.0],
            [0.0, 0.0, 0.85, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
          ],
          type: :f64
        )

      model =
        SpectralClustering.fit(affinity, num_clusters: 3, affinity: :precomputed, key: key())

      assert_same_partition(model.labels, [0, 0, 1, 1, 2])
      refute model.embedding |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 1
    end

    test "non-convex clusters: two nested half-moons are separated" do
      # k-means alone cannot separate these two arcs; spectral clustering can.
      outer =
        for i <- 0..9 do
          angle = :math.pi() * i / 9
          [:math.cos(angle), :math.sin(angle)]
        end

      inner =
        for i <- 0..9 do
          angle = :math.pi() * i / 9
          [1.0 - :math.cos(angle), 0.5 - :math.sin(angle)]
        end

      x = Nx.tensor(outer ++ inner, type: :f64)
      model = SpectralClustering.fit(x, num_clusters: 2, gamma: 20.0, key: key())

      expected = List.duplicate(0, 10) ++ List.duplicate(1, 10)
      assert_same_partition(model.labels, expected)
    end
  end

  describe "properties" do
    test "returns f32 embedding and integer labels for f32 input" do
      x = Nx.tensor([[0.0, 0.0], [0.1, 0.1], [3.0, 3.0], [3.1, 3.1]])
      model = SpectralClustering.fit(x, num_clusters: 2, key: key())

      assert Nx.type(model.embedding) == {:f, 32}
      assert Nx.type(model.labels) == {:s, 32}
      assert Nx.shape(model.labels) == {4}
      assert Nx.shape(model.embedding) == {4, 2}
    end

    test "works inside jit" do
      x =
        Nx.tensor(
          [
            [0.0, 0.0],
            [0.3, 0.2],
            [0.2, 0.4],
            [0.4, 0.1],
            [1.5, 1.4],
            [1.7, 1.6],
            [1.6, 1.8],
            [1.8, 1.5]
          ],
          type: :f64
        )

      k = key()
      direct = SpectralClustering.fit(x, num_clusters: 2, key: k)

      jitted =
        Nx.Defn.jit_apply(
          &SpectralClustering.fit(&1, num_clusters: 2, key: k),
          [x]
        )

      assert Nx.to_flat_list(jitted.labels) == Nx.to_flat_list(direct.labels)
      assert_all_close(jitted.embedding, direct.embedding, atol: 1.0e-12)
    end

    test "num_clusters == num_samples assigns every point its own cluster" do
      x = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], type: :f64)
      model = SpectralClustering.fit(x, num_clusters: 4, key: key())

      assert model.labels |> Nx.sort() |> Nx.to_flat_list() == [0, 1, 2, 3]
    end
  end

  describe "errors" do
    test "raises for a non-rank-2 input" do
      assert_raise ArgumentError,
                   "expected input tensor to have shape {num_samples, num_features} or " <>
                     "{num_samples, num_samples}, got tensor with shape: {3}",
                   fn ->
                     SpectralClustering.fit(Nx.tensor([1, 2, 3]), num_clusters: 2)
                   end
    end

    test "raises for fewer than two samples" do
      assert_raise ArgumentError, "expected at least 2 samples, got: 1", fn ->
        SpectralClustering.fit(Nx.tensor([[3.0, 4.0]]), num_clusters: 1)
      end
    end

    test "raises for a non-square precomputed affinity" do
      assert_raise ArgumentError,
                   "expected a square affinity matrix for affinity: :precomputed, " <>
                     "got tensor with shape: {3, 2}",
                   fn ->
                     SpectralClustering.fit(Nx.broadcast(1.0, {3, 2}),
                       num_clusters: 2,
                       affinity: :precomputed
                     )
                   end
    end

    test "raises when num_clusters exceeds the number of samples" do
      assert_raise ArgumentError,
                   "invalid value for :num_clusters option: expected positive integer " <>
                     "between 1 and 3, got: 4",
                   fn ->
                     SpectralClustering.fit(Nx.broadcast(1.0, {3, 2}), num_clusters: 4)
                   end
    end
  end
end
