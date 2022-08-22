defmodule Scholar.Cluster.KMeansTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  # Reorders clusters according to the first coordinate
  defnp sort_clusters(clusters, labels) do
    order =
      Nx.argsort(clusters[0..-1//1], axis: 0)
      |> Nx.slice_along_axis(0, 1, axis: 1)
      |> Nx.squeeze()

    labels_maping = Nx.argsort(order)
    {Nx.take(clusters, order), Nx.take(labels_maping, labels)}
  end

  test "without weights" do
    model =
      Scholar.Cluster.KMeans.fit(Nx.tensor([[1, 2], [2, 4], [1, 3], [2, 5]]),
        num_clusters: 2
      )

    {oredered_clusters, ordered_labels} = sort_clusters(model.clusters, model.labels)
    model = %{model | :labels => ordered_labels}
    model = %{model | :clusters => oredered_clusters}

    assert model.clusters == Nx.tensor([[1.0, 2.5], [2.0, 4.5]])
    assert model.inertia == Nx.tensor(1.0, type: {:f, 32})
    assert model.labels == Nx.tensor([0, 1, 0, 1])
    assert model.num_iterations == Nx.tensor(2)

    predictions = Scholar.Cluster.KMeans.predict(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
    assert predictions == Nx.tensor([1, 0])
  end

  test "with weights" do
    model =
      Scholar.Cluster.KMeans.fit(Nx.tensor([[1, 2], [2, 4.25], [1, 3], [2, 5]]),
        num_clusters: 2,
        weights: [1, 2, 3, 4]
      )

    {oredered_clusters, ordered_labels} = sort_clusters(model.clusters, model.labels)
    model = %{model | :labels => ordered_labels}
    model = %{model | :clusters => oredered_clusters}

    assert model.clusters == Nx.tensor([[1.0, 2.75], [2.0, 4.75]])
    assert model.inertia == Nx.tensor(1.5, type: {:f, 32})
    assert model.labels == Nx.tensor([0, 1, 0, 1])
    assert model.num_iterations == Nx.tensor(2)

    predictions = Scholar.Cluster.KMeans.predict(model, Nx.tensor([[1.9, 4.3], [1.1, 2.0]]))
    assert predictions == Nx.tensor([1, 0])
  end

  describe "errors" do
    test "when :num_clusters is not provided" do
      x = Nx.tensor([[1, 2], [3, 4], [5, 6]])

      assert_raise ArgumentError,
                   "missing option :num_clusters",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x)
                   end
    end

    test "when :num_clusters is invalid" do
      x = Nx.tensor([[1, 2], [3, 4], [5, 6]])

      assert_raise ArgumentError,
                   "expected :num_clusters to to be a positive integer in range 1 to 3, got: 4",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 4)
                   end

      assert_raise ArgumentError,
                   "expected :num_clusters to to be a positive integer in range 1 to 3, got: 2.0",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 2.0)
                   end

      assert_raise ArgumentError,
                   "expected :num_clusters to to be a positive integer in range 1 to 3, got: -1",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: -1)
                   end
    end

    test "when training vector size is invalid" do
      x = Nx.tensor([5, 6])

      assert_raise ArgumentError,
                   "expected x to have shape {n_samples, n_features}, got tensor with shape: {2}",
                   fn -> Scholar.Cluster.KMeans.fit(x, num_clusters: 2) end
    end

    test "when :max_iterations is not a positive integer" do
      x = Nx.tensor([[1, 2], [3, 4]])

      assert_raise ArgumentError,
                   "expected :max_iterations to be a positive integer, got: 0",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 2, max_iterations: 0)
                   end

      assert_raise ArgumentError,
                   "expected :max_iterations to be a positive integer, got: 200.0",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 2, max_iterations: 200.0)
                   end
    end

    test "when :num_runs is not a positive integer" do
      x = Nx.tensor([[1, 2], [3, 4]])

      assert_raise ArgumentError,
                   "expected :num_runs to be a positive integer, got: 0",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 2, num_runs: 0)
                   end

      assert_raise ArgumentError,
                   "expected :num_runs to be a positive integer, got: 10.0",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 2, num_runs: 10.0)
                   end
    end

    test "when :tol is not a non-negative number" do
      x = Nx.tensor([[1, 2], [3, 4]])

      assert_raise ArgumentError,
                   "expected :tol to be a non-negative number, got: -0.1",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 2, tol: -0.1)
                   end
    end

    test "when :init is neither :random nor a :k_means_plus_plus" do
      x = Nx.tensor([[1, 2], [3, 4]])

      assert_raise ArgumentError,
                   "expected :init to be either :random or :k_means_plus_plus, got: :abc",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 2, init: :abc)
                   end
    end

    test "when :weights is not a list of positive numbers" do
      x = Nx.tensor([[1, 2], [3, 4]])

      assert_raise ArgumentError,
                   "expected :weights to be a list of positive numbers of size 2, got: [1, 2, 3]",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 2, weights: [1, 2, 3])
                   end

      assert_raise ArgumentError,
                   "expected :weights to be a list of positive numbers of size 2, got: [1, -2]",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 2, weights: [1, -2])
                   end

      assert_raise ArgumentError,
                   "expected :weights to be a list of positive numbers of size 2, got: {1, 2}",
                   fn ->
                     Scholar.Cluster.KMeans.fit(x, num_clusters: 2, weights: {1, 2})
                   end
    end
  end
end
