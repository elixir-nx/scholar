defmodule Scholar.Cluster.GaussianMixtureTest do
  use Scholar.Case, async: true
  alias Scholar.Cluster.GaussianMixture
  doctest GaussianMixture

  describe "invalid arguments" do
    test "when :num_gaussians is not provided" do
      x = Nx.tensor([[1, 2], [3, 4], [5, 6]])

      assert_raise NimbleOptions.ValidationError,
                   "required :num_gaussians option not found, received options: []",
                   fn ->
                     GaussianMixture.fit(x)
                   end
    end

    test "when :num_gaussians is invalid" do
      x = Nx.tensor([[1, 2], [3, 4], [5, 6]])

      assert_raise ArgumentError,
                   "invalid value for :num_gaussians option: expected positive integer between 1 and 3, got: 4",
                   fn ->
                     GaussianMixture.fit(x, num_gaussians: 4)
                   end

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :num_gaussians option: expected positive integer, got: 2.0",
                   fn ->
                     GaussianMixture.fit(x, num_gaussians: 2.0)
                   end

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :num_gaussians option: expected positive integer, got: -1",
                   fn ->
                     GaussianMixture.fit(x, num_gaussians: -1)
                   end
    end

    test "when training vector size is invalid" do
      x = Nx.tensor([5, 6])

      assert_raise ArgumentError,
                   "expected input tensor to have shape {n_samples, n_features}, got tensor with shape: {2}",
                   fn -> GaussianMixture.fit(x, num_gaussians: 2) end
    end

    test "when :num_runs is not a positive integer" do
      x = Nx.tensor([[1, 2], [3, 4]])

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :num_runs option: expected positive integer, got: 0",
                   fn ->
                     GaussianMixture.fit(x, num_gaussians: 2, num_runs: 0)
                   end

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :num_runs option: expected positive integer, got: 10.0",
                   fn ->
                     GaussianMixture.fit(x, num_gaussians: 2, num_runs: 10.0)
                   end
    end

    test "when :max_iter is not a positive integer" do
      x = Nx.tensor([[1, 2], [3, 4]])

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :max_iter option: expected positive integer, got: 0",
                   fn ->
                     GaussianMixture.fit(x, num_gaussians: 2, max_iter: 0)
                   end

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :max_iter option: expected positive integer, got: 200.0",
                   fn ->
                     GaussianMixture.fit(x, num_gaussians: 2, max_iter: 200.0)
                   end
    end

    test "when :tol is not a non-negative number" do
      x = Nx.tensor([[1, 2], [3, 4]])

      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :tol option: expected a non-negative number, got: -0.1",
                   fn ->
                     GaussianMixture.fit(x, num_gaussians: 2, tol: -0.1)
                   end
    end
  end
end
