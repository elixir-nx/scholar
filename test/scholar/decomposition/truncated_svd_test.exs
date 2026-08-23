defmodule Scholar.Decomposition.TruncatedSVDTest do
  use Scholar.Case, async: true
  alias Scholar.Decomposition.TruncatedSVD
  doctest TruncatedSVD

  defp key do
    Nx.Random.key(1)
  end

  test "fit test - all default options" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(
        key,
        Nx.tensor([0.0, 0.0, 0.0, 0.0]),
        Nx.tensor([
          [3.0, 2.0, 1.0, 9.0],
          [1.0, 2.0, 3.0, 8.2],
          [1.3, 1.0, 2.2, 2.4],
          [1.8, 1.0, 2.0, 2.9]
        ]),
        shape: {50},
        type: :f32
      )

    model = Scholar.Decomposition.TruncatedSVD.fit(x, key: key)

    assert_all_close(
      model.components,
      Nx.tensor([
        [0.49886093, 0.4450935, 0.50565857, 0.54528815],
        [0.45053825, 0.5961276, -0.51091313, -0.42498812]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.explained_variance,
      Nx.tensor([5.641444, 1.3302106]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.explained_variance_ratio,
      Nx.tensor([0.6498977, 0.15324104]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.explained_variance_ratio,
      Nx.tensor([0.6498977, 0.15324104]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.singular_values,
      Nx.tensor([16.818216, 8.336306]),
      atol: 1.0e-3
    )
  end

  test "fit_transform test - all default options" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(
        key,
        Nx.tensor([0.0, 0.0, 0.0, 0.0]),
        Nx.tensor([
          [3.0, 2.0, 1.0, 9.0],
          [1.0, 2.0, 3.0, 8.2],
          [1.3, 1.0, 2.2, 2.4],
          [1.8, 1.0, 2.0, 2.9]
        ]),
        shape: {10},
        type: :f32
      )

    x_reduced = Scholar.Decomposition.TruncatedSVD.fit_transform(x, key: key)

    assert_all_close(
      x_reduced,
      Nx.tensor([
        [4.44003, -1.567811],
        [-2.1890442, -1.2367431],
        [-0.9577365, -1.4795241],
        [2.227599, 0.14134224],
        [2.8790033, -0.12000961],
        [2.849527, 0.8297561],
        [1.9480042, 0.96068],
        [2.1394887, -1.0573206],
        [-1.2661155, -0.7540298],
        [-0.883705, 0.06961638]
      ]),
      atol: 1.0e-3
    )
  end

  test "fit_transform test - :num_components" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(
        key,
        Nx.tensor([0.0, 0.0, 0.0, 0.0]),
        Nx.tensor([
          [3.0, 2.0, 1.0, 9.0],
          [1.0, 2.0, 3.0, 8.2],
          [1.3, 1.0, 2.2, 2.4],
          [1.8, 1.0, 2.0, 2.9]
        ]),
        shape: {10},
        type: :f32
      )

    x_reduced = Scholar.Decomposition.TruncatedSVD.fit_transform(x, key: key, num_components: 3)

    assert_all_close(
      x_reduced,
      Nx.tensor([
        [4.440031, -1.5678102, 0.08212819],
        [-2.1890433, -1.2367446, 1.2187406],
        [-0.9577367, -1.4795232, -0.5904436],
        [2.2275999, 0.14134066, 0.8389634],
        [2.8790033, -0.120009124, -0.69796675],
        [2.8495266, 0.82975703, -0.13372375],
        [1.948004, 0.96068144, 0.5970833],
        [2.139489, -1.0573193, 0.2925225],
        [-1.2661155, -0.75403, -0.5198247],
        [-0.8837051, 0.06961634, 0.21251045]
      ]),
      atol: 1.0e-3
    )
  end

  test "fit_transform test - :num_oversamples" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(
        key,
        Nx.tensor([0.0, 0.0, 0.0, 0.0]),
        Nx.tensor([
          [3.0, 2.0, 1.0, 9.0],
          [1.0, 2.0, 3.0, 8.2],
          [1.3, 1.0, 2.2, 2.4],
          [1.8, 1.0, 2.0, 2.9]
        ]),
        shape: {10},
        type: :f32
      )

    x_reduced = Scholar.Decomposition.TruncatedSVD.fit_transform(x, key: key, num_oversamples: 20)

    assert_all_close(
      x_reduced,
      Nx.tensor([
        [4.4400306, -1.5678108],
        [-2.1890442, -1.2367437],
        [-0.9577366, -1.4795235],
        [2.2275991, 0.14134149],
        [2.8790035, -0.120009415],
        [2.8495271, 0.82975626],
        [1.9480042, 0.9606803],
        [2.1394887, -1.0573202],
        [-1.2661155, -0.7540297],
        [-0.8837051, 0.069616355]
      ]),
      atol: 1.0e-3
    )
  end

  test "fit_transform test - :num_iters" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(
        key,
        Nx.tensor([0.0, 0.0, 0.0, 0.0]),
        Nx.tensor([
          [3.0, 2.0, 1.0, 9.0],
          [1.0, 2.0, 3.0, 8.2],
          [1.3, 1.0, 2.2, 2.4],
          [1.8, 1.0, 2.0, 2.9]
        ]),
        shape: {10},
        type: :f32
      )

    x_reduced = Scholar.Decomposition.TruncatedSVD.fit_transform(x, key: key, num_iter: 20)

    assert_all_close(
      x_reduced,
      Nx.tensor([
        [4.44003, -1.567811],
        [-2.1890442, -1.2367435],
        [-0.95773673, -1.4795235],
        [2.2275991, 0.14134137],
        [2.8790033, -0.12000972],
        [2.8495271, 0.8297562],
        [1.9480044, 0.9606804],
        [2.1394887, -1.0573201],
        [-1.2661157, -0.75402975],
        [-0.8837051, 0.06961647]
      ]),
      atol: 1.0e-3
    )
  end
end
