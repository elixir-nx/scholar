defmodule Scholar.Cluster.BIRCHTest do
  use Scholar.Case, async: true
  alias Scholar.Cluster.BIRCH
  doctest BIRCH

  defp key, do: Nx.Random.key(42)

  defp x do
    Nx.tensor([
      [1, 2],
      [1, 3],
      [2, 2],
      [8, 8],
      [9, 8],
      [8, 9]
    ])
  end

  # Three well-separated blobs of 8 points each, rounded to 3 decimals so the same
  # values can be fed to scikit-learn.
  defp blobs do
    Nx.tensor([
      [0.706, 0.16],
      [0.391, 0.896],
      [0.747, -0.391],
      [0.38, -0.061],
      [-0.041, 0.164],
      [0.058, 0.582],
      [0.304, 0.049],
      [0.178, 0.133],
      [5.598, 4.918],
      [5.125, 4.658],
      [3.979, 5.261],
      [5.346, 4.703],
      [5.908, 4.418],
      [5.018, 4.925],
      [5.613, 5.588],
      [5.062, 5.151],
      [-0.355, 4.208],
      [-0.139, 5.063],
      [0.492, 5.481],
      [-0.155, 4.879],
      [-0.419, 4.432],
      [-0.683, 5.78],
      [-0.204, 4.825],
      [-0.501, 5.311]
    ])
  end

  # Leaf subclusters are collected in tree order, which need not match the order
  # scikit-learn walks its leaves in, so centers are compared as a set.
  defp sorted_rows(tensor), do: tensor |> Nx.to_list() |> Enum.sort() |> Nx.tensor()

  # Compares two label assignments as partitions, i.e. up to a permutation of the
  # label ids (which are arbitrary for clustering algorithms).
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

  describe "CF-tree" do
    # The CF-tree is built the same way regardless of the global clustering step, so
    # the leaf subclusters can be compared against scikit-learn value by value.
    # Reference values from sklearn 1.9.0 (`Birch(threshold=..., branching_factor=...)`).
    test "leaf subclusters match sklearn" do
      model = BIRCH.fit(blobs(), num_clusters: 3, threshold: 0.5, key: key())

      assert_all_close(
        sorted_rows(model.subcluster_centers),
        Nx.tensor([
          [-0.592, 5.5455],
          [-0.2544, 4.6814],
          [0.2584, 0.387],
          [0.477, -0.134333],
          [0.492, 5.481],
          [3.979, 5.261],
          [5.381429, 4.908714]
        ])
      )
    end

    # branching_factor: 3 overflows the nodes, so this exercises node splitting and
    # the growth of a new root. sklearn produces the same leaves as it does at the
    # default branching factor, and so must we.
    test "leaf subclusters match sklearn when nodes split" do
      model = BIRCH.fit(blobs(), num_clusters: 3, threshold: 0.5, branching_factor: 3, key: key())

      assert_all_close(
        sorted_rows(model.subcluster_centers),
        Nx.tensor([
          [-0.592, 5.5455],
          [-0.2544, 4.6814],
          [0.2584, 0.387],
          [0.477, -0.134333],
          [0.492, 5.481],
          [3.979, 5.261],
          [5.381429, 4.908714]
        ])
      )
    end

    # When a node splits, the first half has to replace the old subcluster in place and
    # the second half has to go to the end of the node. Subcluster order decides which
    # entry wins a distance tie on later descents, so getting it wrong changes the tree.
    # This dataset sits on an integer grid at branching_factor: 2, so it is full of ties
    # and nested splits; inserting both halves at the original position fails it.
    test "leaf subclusters match sklearn when a split reorders a node" do
      data =
        Nx.tensor([
          [3, 1],
          [3, 3],
          [4, 1],
          [1, 0],
          [2, 1],
          [0, 1],
          [1, 0],
          [2, 4],
          [4, 4],
          [4, 2],
          [2, 4],
          [3, 3],
          [0, 4],
          [2, 4],
          [4, 1]
        ])

      model = BIRCH.fit(data, num_clusters: 5, threshold: 1.0, branching_factor: 2, key: key())

      assert_all_close(
        sorted_rows(model.subcluster_centers),
        Nx.tensor([
          [0.0, 4.0],
          [1.0, 0.5],
          [2.6, 3.8],
          [3.333333, 2.0],
          [4.0, 1.0]
        ])
      )
    end

    test "a larger threshold merges more points into each subcluster" do
      model = BIRCH.fit(blobs(), num_clusters: 3, threshold: 1.0, branching_factor: 3, key: key())

      assert_all_close(
        sorted_rows(model.subcluster_centers),
        Nx.tensor([
          [-0.2455, 4.997375],
          [0.340375, 0.1915],
          [5.206125, 4.95275]
        ])
      )
    end
  end

  describe "fit" do
    test "matches sklearn on three well-separated blobs" do
      model = BIRCH.fit(blobs(), num_clusters: 3, key: key())

      assert_same_partition(
        model.labels,
        List.duplicate(1, 8) ++ List.duplicate(0, 8) ++ List.duplicate(2, 8)
      )
    end

    test "separates two well-defined blobs" do
      model = BIRCH.fit(x(), num_clusters: 2, key: key())

      assert_same_partition(model.labels, [0, 0, 0, 1, 1, 1])
    end

    test "handles nodes that split" do
      {data, _} = Nx.Random.normal(key(), 0.0, 1.0, shape: {50, 2})
      model = BIRCH.fit(data, num_clusters: 3, branching_factor: 3, threshold: 0.3, key: key())

      assert Nx.shape(model.labels) == {50}
      assert Nx.axis_size(model.subcluster_centers, 0) > 3
    end

    test "labels every point" do
      model = BIRCH.fit(x(), num_clusters: 2, key: key())

      assert Nx.shape(model.labels) == {6}
      assert Nx.axis_size(model.subcluster_centers, 1) == 2
    end

    test "num_clusters not exceeding subcluster count gives each subcluster its own label" do
      # A tight threshold keeps every distinct point in its own leaf subcluster, and
      # num_clusters >= that count means no global merging happens.
      model = BIRCH.fit(x(), threshold: 0.1, num_clusters: 6, key: key())
      centers = model.subcluster_centers

      assert model.subcluster_labels == Nx.iota({Nx.axis_size(centers, 0)}, type: :s32)
    end

    test "the same key gives the same clustering" do
      opts = [num_clusters: 3, branching_factor: 3, threshold: 0.5, key: key()]

      assert BIRCH.fit(blobs(), opts).labels == BIRCH.fit(blobs(), opts).labels
    end

    test "accepts integer input" do
      assert BIRCH.fit(x(), num_clusters: 2, key: key()).labels ==
               BIRCH.fit(Nx.as_type(x(), :f32), num_clusters: 2, key: key()).labels
    end
  end

  describe "errors" do
    test "when the training vector is not a matrix" do
      assert_raise ArgumentError,
                   "expected input tensor to have shape {num_samples, num_features}, got tensor with shape: {3}",
                   fn -> BIRCH.fit(Nx.tensor([1, 2, 3]), num_clusters: 2) end
    end

    test "when :num_clusters exceeds the number of samples" do
      assert_raise ArgumentError,
                   "invalid value for :num_clusters option: expected positive integer between 1 and 6, got: 7",
                   fn -> BIRCH.fit(x(), num_clusters: 7) end
    end

    test "when :num_clusters is not a positive integer" do
      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :num_clusters option: expected positive integer, got: 2.0",
                   fn -> BIRCH.fit(x(), num_clusters: 2.0) end
    end

    test "when :threshold is not a positive number" do
      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :threshold option: expected positive number, got: 0",
                   fn -> BIRCH.fit(x(), threshold: 0) end
    end

    test "when :branching_factor is not a positive integer" do
      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :branching_factor option: expected positive integer, got: 0",
                   fn -> BIRCH.fit(x(), branching_factor: 0) end
    end
  end

  describe "predict" do
    test "assigns new points to the nearest cluster" do
      model = BIRCH.fit(x(), num_clusters: 2, key: key())
      preds = BIRCH.predict(model, Nx.tensor([[1.5, 2.5], [8.5, 8.5]]))

      near_first = model.labels[0] |> Nx.to_number()
      near_second = model.labels[3] |> Nx.to_number()

      assert preds == Nx.tensor([near_first, near_second], type: :s32)
    end

    test "raises when the number of features does not match" do
      model = BIRCH.fit(x(), num_clusters: 2, key: key())

      assert_raise ArgumentError,
                   "expected tensor to have shape {3}, got tensor with shape {2}",
                   fn -> BIRCH.predict(model, Nx.tensor([[1.0, 2.0, 3.0]])) end
    end
  end
end
