defmodule Scholar.Cluster.HDBSCANTest do
  use Scholar.Case, async: true
  alias Scholar.Cluster.HDBSCAN
  doctest HDBSCAN

  # Three well separated groups of six. Chosen so that every merge height in the single
  # linkage tree over the mutual reachability is distinct, which makes the dendrogram
  # unique and therefore forces any correct implementation to the same answer. That is
  # what lets the assertions below compare labels exactly rather than up to renumbering.
  defp blobs do
    Nx.tensor([
      [1.15, 1.585],
      [1.834, 0.694],
      [1.915, 1.287],
      [1.525, -0.246],
      [1.64, 0.51],
      [0.822, 0.832],
      [8.806, 2.281],
      [9.793, 1.606],
      [9.14, 2.101],
      [8.238, 1.596],
      [8.866, 1.012],
      [9.682, 2.307],
      [5.328, 7.954],
      [5.26, 8.128],
      [5.631, 9.082],
      [5.008, 8.702],
      [4.941, 8.269],
      [5.978, 9.37]
    ])
  end

  describe "fit" do
    # Reference: sklearn.cluster.HDBSCAN(min_cluster_size=3, min_samples=3).fit(x).labels_
    test "recovers three separated blobs" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 3)

      assert model.labels ==
               Nx.tensor([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], type: :s32)
    end

    # Reference: sklearn.cluster.HDBSCAN(min_cluster_size=4, min_samples=4).fit(x).labels_
    test "a larger min_cluster_size keeps the same three blobs" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 4, min_samples: 4)

      assert model.labels ==
               Nx.tensor([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], type: :s32)
    end

    test "a min_cluster_size larger than any blob leaves only noise" do
      model = HDBSCAN.fit(blobs(), min_cluster_size: 7, min_samples: 3)

      assert model.labels == Nx.broadcast(Nx.tensor(-1, type: :s32), {18})
    end

    test "labels are contiguous from zero, with -1 reserved for noise" do
      for min_cluster_size <- 2..7 do
        labels =
          blobs()
          |> HDBSCAN.fit(min_cluster_size: min_cluster_size, min_samples: 3)
          |> Map.fetch!(:labels)
          |> Nx.to_flat_list()

        clusters = labels |> Enum.reject(&(&1 == -1)) |> Enum.uniq() |> Enum.sort()

        assert Enum.all?(labels, &(&1 >= -1))
        assert clusters == Enum.to_list(0..(length(clusters) - 1)//1)
      end
    end

    test "min_samples defaults to min_cluster_size" do
      assert HDBSCAN.fit(blobs(), min_cluster_size: 4) ==
               HDBSCAN.fit(blobs(), min_cluster_size: 4, min_samples: 4)
    end

    test "min_samples changes the result" do
      loose = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 2)
      tight = HDBSCAN.fit(blobs(), min_cluster_size: 3, min_samples: 12)

      refute loose.labels == tight.labels
    end
  end
end
