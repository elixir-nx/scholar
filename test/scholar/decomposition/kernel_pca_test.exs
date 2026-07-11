defmodule Scholar.Decomposition.KernelPCATest do
  use Scholar.Case, async: true
  alias Scholar.Decomposition.KernelPCA
  doctest KernelPCA

  defp x do
    Nx.tensor([
      [0.5, 0.2, 0.8],
      [1.0, 0.5, 0.2],
      [0.3, 1.0, 0.7],
      [0.9, 0.1, 1.0],
      [0.4, 0.8, 0.3],
      [0.6, 0.2, 0.9]
    ])
  end

  defp x_test do
    Nx.tensor([[0.5, 0.5, 0.5], [0.1, 0.9, 0.2]])
  end

  # Reference values taken from scikit-learn (sklearn.decomposition.KernelPCA).
  describe "linear kernel" do
    test "fit/2 eigenvalues" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :linear)
      assert_all_close(model.eigenvalues, Nx.tensor([1.02576507, 0.49150841]))
    end

    test "fit_transform/2" do
      z = KernelPCA.fit_transform(x(), num_components: 2, kernel: :linear)

      assert_all_close(
        z,
        Nx.tensor([
          [0.2480282, -0.14309686],
          [-0.13098757, 0.57609125],
          [-0.49936002, -0.3421911],
          [0.56523378, 0.01348409],
          [-0.5166106, 0.03850971],
          [0.33369622, -0.14279708]
        ]),
        atol: 1.0e-3
      )
    end

    test "transform/2" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :linear)

      assert_all_close(
        KernelPCA.transform(model, x_test()),
        Nx.tensor([[-0.1434428, 0.01749365], [-0.74814817, -0.11777146]]),
        atol: 1.0e-3
      )
    end
  end

  describe "rbf kernel" do
    test "fit_transform/2" do
      z = KernelPCA.fit_transform(x(), num_components: 2, kernel: :rbf)

      assert_all_close(model_eigenvalues(:rbf), Nx.tensor([0.57052273, 0.28089124]))

      assert_all_close(
        z,
        Nx.tensor([
          [0.19920416, -0.10217718],
          [-0.11443324, 0.43767286],
          [-0.3679805, -0.26212427],
          [0.40563184, 0.01272026],
          [-0.38548342, 0.01315336],
          [0.26306115, -0.09924505]
        ]),
        atol: 1.0e-3
      )
    end

    test "transform/2" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :rbf)

      assert_all_close(
        KernelPCA.transform(model, x_test()),
        Nx.tensor([[-0.11224132, 0.01060868], [-0.49994934, -0.10212445]]),
        atol: 1.0e-3
      )
    end
  end

  describe "poly kernel" do
    test "fit_transform/2" do
      z = KernelPCA.fit_transform(x(), num_components: 2, kernel: :poly, degree: 3, coef0: 1.0)

      assert_all_close(
        z,
        Nx.tensor([
          [0.27508957, -0.15923293],
          [-0.16730544, 0.7908652],
          [-0.73099016, -0.47917877],
          [0.85131884, -0.04410387],
          [-0.65531824, 0.08343613],
          [0.42720543, -0.19178576]
        ]),
        atol: 1.0e-3
      )
    end
  end

  describe "sigmoid kernel" do
    test "fit_transform/2" do
      z = KernelPCA.fit_transform(x(), num_components: 2, kernel: :sigmoid, coef0: 1.0)

      assert_all_close(
        z,
        Nx.tensor([
          [-0.08427136, -0.04391895],
          [0.04163971, 0.15793664],
          [0.12574582, -0.09492698],
          [-0.13989375, 0.01645784],
          [0.15586995, 0.00172166],
          [-0.09909037, -0.03727021]
        ]),
        atol: 1.0e-3
      )
    end
  end

  describe "cosine kernel" do
    test "fit_transform/2" do
      z = KernelPCA.fit_transform(x(), num_components: 2, kernel: :cosine)

      assert_all_close(
        z,
        Nx.tensor([
          [-0.30958794, -0.13265305],
          [0.1569742, 0.49544337],
          [0.37577318, -0.30154498],
          [-0.40180281, 0.06364262],
          [0.50860284, -0.02523757],
          [-0.32995947, -0.09965038]
        ]),
        atol: 1.0e-3
      )
    end
  end

  describe "general behaviour" do
    test "fit_transform/2 equals fit/2 followed by transform/2 on training data" do
      for kernel <- [:linear, :poly, :rbf, :sigmoid, :cosine] do
        opts = [num_components: 2, kernel: kernel]
        model = KernelPCA.fit(x(), opts)

        assert_all_close(
          KernelPCA.fit_transform(x(), opts),
          KernelPCA.transform(model, x()),
          atol: 1.0e-3
        )
      end
    end

    test "the number of components sets the output dimension" do
      z = KernelPCA.fit_transform(x(), num_components: 3, kernel: :rbf)
      assert Nx.shape(z) == {6, 3}
    end

    test "fit/2 and transform/2 work with jit_apply" do
      model = Nx.Defn.jit_apply(&KernelPCA.fit(&1, num_components: 2, kernel: :rbf), [x()])
      z = Nx.Defn.jit_apply(&KernelPCA.transform/2, [model, x_test()])
      assert Nx.shape(z) == {2, 2}
    end

    test "fit/2 propagates input precision (f64)" do
      x = Nx.as_type(x(), :f64)
      model = KernelPCA.fit(x, num_components: 2, kernel: :rbf)
      assert Nx.type(model.eigenvalues) == {:f, 64}
      assert Nx.type(KernelPCA.transform(model, Nx.as_type(x_test(), :f64))) == {:f, 64}
    end
  end

  describe "input validation" do
    test "fit/2 requires a rank-2 tensor" do
      assert_raise ArgumentError,
                   ~r/expected input tensor to have shape \{num_samples, num_features\}/,
                   fn -> KernelPCA.fit(Nx.iota({5}), num_components: 2) end
    end

    test "num_components may not exceed num_samples" do
      assert_raise ArgumentError,
                   "num_components must be less than or equal to num_samples = 6, got 7",
                   fn -> KernelPCA.fit(x(), num_components: 7) end
    end
  end

  defp model_eigenvalues(kernel) do
    KernelPCA.fit(x(), num_components: 2, kernel: kernel).eigenvalues
  end
end
