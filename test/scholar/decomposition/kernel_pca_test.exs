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

    test "fit/2 eigenvalues" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :poly, degree: 3, coef0: 1.0)
      assert_all_close(model.eigenvalues, Nx.tensor([1.97470225, 0.9261237]))
    end

    test "transform/2" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :poly, degree: 3, coef0: 1.0)

      assert_all_close(
        KernelPCA.transform(model, x_test()),
        Nx.tensor([[-0.21107767, 0.05450511], [-0.87419523, -0.07912082]]),
        atol: 1.0e-3
      )
    end

    test "custom degree" do
      z = KernelPCA.fit_transform(x(), num_components: 2, kernel: :poly, degree: 2, coef0: 1.0)

      assert_all_close(
        z,
        Nx.tensor([
          [0.21428251, -0.12551537],
          [-0.11978673, 0.55138703],
          [-0.49389717, -0.32901762],
          [0.56577048, -0.00833521],
          [-0.47484395, 0.04791742],
          [0.30847487, -0.13643626]
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

    test "fit/2 eigenvalues" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :sigmoid, coef0: 1.0)
      assert_all_close(model.eigenvalues, Nx.tensor([0.07833214, 0.03754688]))
    end

    test "transform/2" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :sigmoid, coef0: 1.0)

      assert_all_close(
        KernelPCA.transform(model, x_test()),
        Nx.tensor([[0.03661575, -0.00068769], [0.24706743, -0.05864022]]),
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

    test "fit/2 eigenvalues" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :cosine)
      assert_all_close(model.eigenvalues, Nx.tensor([0.79068667, 0.36860785]))
    end

    test "transform/2" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :cosine)

      assert_all_close(
        KernelPCA.transform(model, x_test()),
        Nx.tensor([[0.11596968, -0.0139805], [0.72969734, -0.22251837]]),
        atol: 1.0e-3
      )
    end

    test "rows with zero norm do not produce NaN" do
      x_with_zero_row =
        Nx.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 0.2], [0.3, 1.0, 0.7], [0.9, 0.1, 1.0]])

      model = KernelPCA.fit(x_with_zero_row, num_components: 2, kernel: :cosine)

      # Reference values taken from scikit-learn, which treats zero-norm rows
      # as an all-zero vector instead of dividing by zero.
      assert_all_close(model.eigenvalues, Nx.tensor([0.59310028, 0.38942085]))

      assert_all_close(
        KernelPCA.fit_transform(x_with_zero_row, num_components: 2, kernel: :cosine),
        Nx.tensor([
          [0.66198055, -0.06451232],
          [-0.2645941, -0.1783384],
          [-0.14431492, 0.52389659],
          [-0.25307153, -0.28104588]
        ]),
        atol: 1.0e-3
      )
    end
  end

  describe "rbf kernel with custom gamma" do
    test "fit_transform/2" do
      z = KernelPCA.fit_transform(x(), num_components: 2, kernel: :rbf, gamma: 2.0)

      assert_all_close(
        z,
        Nx.tensor([
          [-0.43668454, -0.12918044],
          [0.29479921, 0.79022216],
          [0.5971787, -0.46764162],
          [-0.56175187, 0.01788282],
          [0.63960914, -0.09610005],
          [-0.53315064, -0.11518287]
        ]),
        atol: 1.0e-3
      )
    end

    test "fit/2 keeps the given gamma instead of the default" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :rbf, gamma: 2.0)
      assert model.gamma == 2.0
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

      transformed = KernelPCA.transform(model, Nx.as_type(x_test(), :f64))
      assert Nx.type(transformed) == {:f, 64}

      # f64 gets close to the f64 SciPy/scikit-learn reference
      assert_all_close(
        transformed,
        Nx.tensor([[-0.11224132, 0.01060868], [-0.49994934, -0.10212445]], type: :f64),
        atol: 1.0e-5
      )
    end

    test "kernel defaults to :linear" do
      assert KernelPCA.fit(x(), num_components: 2).kernel == :linear
    end

    test "gamma defaults to 1 / num_features" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :rbf)
      assert_all_close(Nx.tensor(model.gamma), Nx.tensor(1.0 / 3))
    end

    test "num_components may equal num_samples" do
      # The centering forces one zero eigenvalue (up to floating-point noise,
      # which can make it slightly negative), so this exercises the clipping
      # in fit/2 and the zero-eigenvalue guard in transform/2.
      model = KernelPCA.fit(x(), num_components: 6, kernel: :rbf)
      assert Nx.shape(model.eigenvalues) == {6}
      assert Nx.to_number(model.eigenvalues[5]) == 0.0

      z = KernelPCA.fit_transform(x(), num_components: 6, kernel: :rbf)
      assert Nx.to_number(Nx.any(Nx.is_nan(z))) == 0

      zt = KernelPCA.transform(model, x_test())
      assert Nx.to_number(Nx.any(Nx.is_nan(zt))) == 0
      assert Nx.to_number(Nx.any(Nx.is_infinity(zt))) == 0

      # the zero-eigenvalue component projects to zero, as in scikit-learn
      assert zt[[.., 5]] == Nx.tensor([0.0, 0.0])
    end

    test "eigenvalues are sorted in decreasing order" do
      model = KernelPCA.fit(x(), num_components: 4, kernel: :rbf)
      sorted = model.eigenvalues |> Nx.to_flat_list() |> Enum.sort(:desc)
      assert Nx.to_flat_list(model.eigenvalues) == sorted
    end

    test "matches scikit-learn on a larger dataset" do
      # On more samples eigh stops before the eigenvalues converge, which the
      # Rayleigh quotient refinement compensates for; without it the second
      # eigenvalue here comes out around 20.73.
      x =
        Nx.tensor(
          [
            [0.1236, 1.8521, 1.196, 0.796],
            [-0.5319, -0.532, -0.8257, 1.5985],
            [0.8033, 1.1242, -0.9382, 1.9097],
            [1.4973, -0.363, -0.4545, -0.4498],
            [-0.0873, 0.5743, 0.2958, -0.1263],
            [0.8356, -0.5815, -0.1236, 0.0991],
            [0.3682, 1.3555, -0.401, 0.5427],
            [0.7772, -0.8606, 0.8226, -0.4884],
            [-0.8048, 1.8467, 1.8969, 1.4252],
            [-0.0862, -0.707, 1.0527, 0.3205],
            [-0.6339, 0.4855, -0.8968, 1.728],
            [-0.2237, 0.9876, -0.0649, 0.5602],
            [0.6401, -0.4454, 1.9088, 1.3254],
            [1.8185, 1.6845, 0.7937, 1.7656],
            [-0.7345, -0.4121, -0.8643, -0.024],
            [0.166, -0.186, 1.4862, 0.0703],
            [-0.1572, 0.6281, -0.5772, 1.4066],
            [-0.7763, 1.9607, 1.3167, -0.4039],
            [-0.9834, 1.4464, 1.1206, 1.187],
            [1.3138, -0.7779, 0.0754, -0.6524],
            [1.5893, 0.8699, -0.0073, -0.8093],
            [-0.0671, -0.0245, 1.1888, 0.9127],
            [1.6616, 0.4166, -0.6412, 1.1397],
            [1.2824, 0.6838, 1.3129, 0.4814],
            [0.5682, 0.2826, -0.9237, -0.6763]
          ],
          type: :f64
        )

      x_test =
        Nx.tensor(
          [
            [-0.9057, 0.9092, -0.0569, 0.5257],
            [1.7227, -0.2521, 0.2311, 1.2667],
            [-0.3136, -0.7691, -0.1307, -0.5163]
          ],
          type: :f64
        )

      model = KernelPCA.fit(x, num_components: 4, kernel: :linear)

      assert_all_close(
        model.eigenvalues,
        Nx.tensor([30.12812417, 20.80703667, 15.30404198, 11.71610252], type: :f64)
      )

      assert_all_close(
        KernelPCA.transform(model, x_test),
        Nx.tensor(
          [
            [0.62235235, 0.56524336, -0.82260582, -0.7056329],
            [-0.72708075, 0.09505719, 0.88921225, 1.27344856],
            [-1.18468312, -0.25757465, -1.27491476, -0.37172622]
          ],
          type: :f64
        )
      )
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

    test "transform/2 requires a rank-2 tensor" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :linear)

      assert_raise ArgumentError,
                   ~r/expected input tensor to have shape \{num_samples, num_features\}/,
                   fn -> KernelPCA.transform(model, Nx.iota({3})) end
    end

    test "transform/2 requires the same number of features used to fit the model" do
      model = KernelPCA.fit(x(), num_components: 2, kernel: :linear)

      assert_raise ArgumentError,
                   "expected input tensor to have the same number of features " <>
                     "as tensor used to fit the model, got 2 and 3",
                   fn -> KernelPCA.transform(model, Nx.tensor([[0.1, 0.2]])) end
    end
  end

  defp model_eigenvalues(kernel) do
    KernelPCA.fit(x(), num_components: 2, kernel: kernel).eigenvalues
  end
end
