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

    tsvd = Scholar.Decomposition.TruncatedSVD.fit(x, key: key)

    assert_all_close(
      model.components,
      Nx.tensor([
        [0.49934840202331543, 0.44504958391189575, 0.5053765773773193, 0.5451390743255615],
        [0.4780271351337433, 0.569697916507721, -0.5178372263908386, -0.42282143235206604]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.explained_variance,
      Nx.tensor([5.641434192657471, 1.3331592082977295]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.explained_variance_ratio,
      Nx.tensor([0.649896502494812, 0.15358072519302368]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.explained_variance_ratio,
      Nx.tensor([0.649896502494812, 0.15358072519302368]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.singular_values,
      Nx.tensor([16.81821060180664, 8.335840225219727]),
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
      model.singular_values,
      Nx.tensor([
        [4.441530227661133, -1.5630521774291992],
        [-2.187946081161499, -1.2309558391571045],
        [-0.9562748074531555, -1.4839725494384766],
        [2.2274107933044434, 0.1483912318944931],
        [2.879176378250122, -0.12200745940208435],
        [2.8487348556518555, 0.8317009806632996],
        [1.9470200538635254, 0.96690434217453],
        [2.140472173690796, -1.0529983043670654],
        [-1.265346884727478, -0.7587057948112488],
        [-0.8837906122207642, 0.07025688886642456]
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
      model.singular_values,
      Nx.tensor([
        [4.441530704498291, -1.5630513429641724, 0.08955635130405426],
        [-2.1879451274871826, -1.2309576272964478, 1.2222723960876465],
        [-0.9562751054763794, -1.4839714765548706, -0.562005341053009],
        [2.2274117469787598, 0.1483895182609558, 0.8012741804122925],
        [2.879176378250122, -0.12200674414634705, -0.7124714255332947],
        [2.8487346172332764, 0.8317020535469055, -0.1308409571647644],
        [1.9470199346542358, 0.9669057130813599, 0.6275887489318848],
        [2.140472412109375, -1.0529969930648804, 0.32528647780418396],
        [-1.2653470039367676, -0.7587059140205383, -0.5229729413986206],
        [-0.8837906122207642, 0.0702567845582962, 0.2195502668619156]
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
      model.singular_values,
      Nx.tensor([
        [4.441530227661133, -1.5630521774291992],
        [-2.187946081161499, -1.2309565544128418],
        [-0.9562748670578003, -1.4839720726013184],
        [2.2274110317230225, 0.14839033782482147],
        [2.879176616668701, -0.12200725078582764],
        [2.8487348556518555, 0.8317012190818787],
        [1.9470199346542358, 0.9669046401977539],
        [2.140472173690796, -1.0529980659484863],
        [-1.265346884727478, -0.7587056756019592],
        [-0.8837906122207642, 0.07025686651468277]
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
      model.singular_values,
      Nx.tensor([
        [4.441530227661133, -1.5630522966384888],
        [-2.18794584274292, -1.2309566736221313],
        [-0.9562749862670898, -1.4839718341827393],
        [2.2274110317230225, 0.1483900398015976],
        [2.879176378250122, -0.1220073327422142],
        [2.8487348556518555, 0.8317012190818787],
        [1.9470200538635254, 0.9669046998023987],
        [2.140472173690796, -1.0529979467391968],
        [-1.265346884727478, -0.7587056159973145],
        [-0.8837906122207642, 0.07025690376758575]
      ]),
      atol: 1.0e-3
    )
  end
end
