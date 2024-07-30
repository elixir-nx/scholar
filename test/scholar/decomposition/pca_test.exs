defmodule Scholar.Decomposition.PCATest do
  use Scholar.Case, async: true
  alias Scholar.Decomposition.PCA
  doctest PCA

  defp x do
    Nx.tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
  end

  defp x2 do
    Nx.tensor([[1, 4], [54, 6], [26, 7]])
  end

  defp x3 do
    Nx.tensor([[-1, -1, 3], [-2, -1, 2], [-3, -2, 1], [3, 1, 1], [21, 2, 1], [5, 3, 2]])
  end

  defp x_wide do
    Nx.tensor([[1, 2, 3], [56, 2, 4]])
  end

  test "fit test - all default options" do
    model = PCA.fit(x(), num_components: 1)

    assert_all_close(
      model.components,
      Nx.tensor([[-0.838727593421936, -0.5445511937141418]]),
      atol: 1.0e-3
    )

    assert_all_close(model.singular_values, Nx.tensor([6.300611972808838]), atol: 1.0e-3)

    assert model.num_samples_seen == Nx.u64(6)

    assert model.mean == Nx.tensor([0.0, 0.0])

    assert_all_close(model.variance, Nx.tensor([5.599999904632568, 2.4000000953674316]),
      atol: 1.0e-3
    )

    assert_all_close(model.explained_variance, Nx.tensor([7.939542293548584]), atol: 1.0e-3)

    assert_all_close(model.explained_variance_ratio, Nx.tensor([0.9924428462982178]))

    assert not model.whiten?
  end

  test "fit test - :num_components is integer and wide matrix" do
    model = PCA.fit(x_wide(), num_components: 1)

    assert_all_close(
      model.components,
      Nx.tensor([
        [-0.9998347759246826, 0.0, -0.01817881315946579]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(model.singular_values, Nx.tensor([38.89730453491211]), atol: 1.0e-3)

    assert model.num_samples_seen == Nx.u64(2)

    assert model.mean == Nx.tensor([28.5, 2.0, 3.5])

    assert model.variance == Nx.tensor([1512.5, 0.0, 0.5])

    assert_all_close(model.explained_variance, Nx.tensor([1513.000244140625]), atol: 1.0e-3)

    assert_all_close(
      model.explained_variance_ratio,
      Nx.tensor([1.0])
    )

    assert not model.whiten?
  end

  test "transform test - :num_components set to 1" do
    model = PCA.fit(x(), num_components: 1)

    assert_all_close(
      PCA.transform(model, x()),
      Nx.tensor([
        [1.3832788467407227],
        [2.222006320953369],
        [3.605285167694092],
        [-1.3832788467407227],
        [-2.222006320953369],
        [-3.605285167694092]
      ]),
      atol: 1.0e-2
    )
  end

  test "transform test - :num_components set to 2" do
    model = PCA.fit(x3(), num_components: 2)

    assert_all_close(
      model.components,
      Nx.tensor([
        [0.9874106645584106, 0.1541961133480072, -0.035266418009996414],
        [-0.14880891144275665, 0.9811362028121948, 0.1234002411365509]
      ]),
      atol: 1.0e-2
    )

    assert_all_close(model.singular_values, Nx.tensor([20.272085189819336, 3.1355254650115967]),
      atol: 1.0e-2
    )

    assert model.num_samples_seen == Nx.u64(6)

    assert_all_close(model.mean, Nx.tensor([3.83333333, 0.33333333, 1.66666667]), atol: 1.0e-2)

    assert_all_close(
      model.variance,
      Nx.tensor([80.16666412353516, 3.866666793823242, 0.6666666865348816]),
      atol: 1.0e-2
    )

    assert_all_close(
      model.explained_variance,
      Nx.tensor([82.19153594970703, 1.966333031654358]),
      atol: 1.0e-2
    )

    assert_all_close(
      model.explained_variance_ratio,
      Nx.tensor([0.97038421, 0.023215265944600105]),
      atol: 1.0e-2
    )

    assert_all_close(
      PCA.transform(model, x3()),
      Nx.tensor([
        [-5.025101184844971, -0.42440494894981384],
        [-5.977245330810547, -0.3989962935447693],
        [-7.083585739135742, -1.3547236919403076],
        [-0.6965337991714478, 0.6958313584327698],
        [17.231054306030273, -1.001592755317688],
        [1.5514134168624878, 2.483886241912842]
      ]),
      atol: 1.0e-2
    )
  end

  test "transform test - :whiten set to false and different data in fit and transform" do
    model = PCA.fit(x(), num_components: 1)

    assert_all_close(
      PCA.transform(model, x2()),
      Nx.tensor([
        [-3.016932487487793],
        [-48.558597564697266],
        [-25.618776321411133]
      ]),
      atol: 1.0e-1,
      rtol: 1.0e-3
    )
  end

  test "transform test - :whiten set to true" do
    model = PCA.fit(x(), num_components: 1, whiten?: true)

    assert_all_close(
      PCA.transform(model, x()),
      Nx.tensor([
        [0.4909214377403259],
        [0.7885832190513611],
        [1.279504656791687],
        [-0.4909214377403259],
        [-0.7885832190513611],
        [-1.279504656791687]
      ]),
      atol: 1.0e-2
    )
  end

  test "fit_transform test - :whiten? set to false and num_components set to 1" do
    model = PCA.fit(x(), num_components: 1)

    assert_all_close(
      PCA.transform(model, x()),
      PCA.fit_transform(x(), num_components: 1),
      atol: 1.0e-2
    )
  end

  test "fit_transform test - :whiten? set to false and num_components set to 2" do
    model = PCA.fit(x3(), num_components: 2)

    assert_all_close(
      PCA.transform(model, x3()),
      PCA.fit_transform(x3(), num_components: 2),
      atol: 1.0e-2
    )
  end

  test "fit_transform test - :whiten? set to true" do
    model = PCA.fit(x(), num_components: 1, whiten?: true)

    assert_all_close(
      PCA.transform(model, x()),
      PCA.fit_transform(x(), num_components: 1, whiten?: true),
      atol: 1.0e-2
    )
  end

  describe "errors" do
    test "input rank different than 2" do
      assert_raise ArgumentError,
                   """
                   expected input tensor to have shape {num_samples, num_features}, \
                   got tensor with shape: {4}\
                   """,
                   fn ->
                     PCA.fit(Nx.tensor([1, 2, 3, 4]), num_components: 1)
                   end
    end

    test "fit test - :num_components bigger than num_features" do
      assert_raise ArgumentError,
                   """
                   num_components must be less than or equal to \
                   num_features = 2, got 4
                   """,
                   fn ->
                     PCA.fit(x(), num_components: 4)
                   end
    end

    test "fit test - :num_components is atom" do
      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :num_components option: expected positive integer, got: :two",
                   fn ->
                     PCA.fit(x(), num_components: :two)
                   end
    end

    test "transform test - :whiten? is not boolean" do
      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :whiten? option: expected boolean, got: :yes",
                   fn ->
                     PCA.fit(x(), num_components: 1, whiten?: :yes)
                   end
    end
  end

  test "partial_fit" do
    model = PCA.fit(x()[0..2], num_components: 1)
    model = PCA.partial_fit(model, x()[3..5])

    assert Nx.shape(model.components) == {1, 2}
    assert Nx.shape(model.singular_values) == {1}
    assert model.num_samples_seen == Nx.u64(6)
    assert model.mean == Nx.tensor([0.0, 0.0])
    assert Nx.shape(model.variance) == {2}
    assert Nx.shape(model.explained_variance) == {1}
    assert Nx.shape(model.explained_variance_ratio) == {1}
    assert not model.whiten?
  end

  test "incremental_fit" do
    batches = Nx.to_batched(x(), 2)
    model = PCA.incremental_fit(batches, num_components: 1)

    assert Nx.shape(model.components) == {1, 2}
    assert Nx.shape(model.singular_values) == {1}
    assert model.num_samples_seen == Nx.u64(6)
    assert model.mean == Nx.tensor([0.0, 0.0])
    assert Nx.shape(model.variance) == {2}
    assert Nx.shape(model.explained_variance) == {1}
    assert Nx.shape(model.explained_variance_ratio) == {1}
    assert not model.whiten?
  end
end
