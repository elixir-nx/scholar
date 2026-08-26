defmodule Scholar.Preprocessing.StandardScalerTest do
  use Scholar.Case, async: true
  alias Scholar.Preprocessing.StandardScaler

  doctest StandardScaler

  describe "fit_transform/2" do
    test "applies standard scaling to data" do
      data = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])

      expected =
        Nx.tensor([
          [0.5212860703468323, -1.3553436994552612, 1.4596009254455566],
          [1.4596009254455566, -0.4170288145542145, -0.4170288145542145],
          [-0.4170288145542145, 0.5212860703468323, -1.3553436994552612]
        ])

      assert_all_close(StandardScaler.fit_transform(data), expected)
    end

    test "centers data around zero when variance is zero" do
      data = 42.0
      expected = Nx.tensor(0.0)
      assert StandardScaler.fit_transform(data) == expected
    end

    test "centers a constant feature to zero while leaving other features scaled normally" do
      data = Nx.tensor([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])

      expected =
        Nx.tensor([
          [-1.2247448, 0.0],
          [0.0, 0.0],
          [1.2247448, 0.0]
        ])

      assert_all_close(StandardScaler.fit_transform(data, axes: [0]), expected)

      scaler = StandardScaler.fit(data, axes: [0])
      assert_all_close(scaler.mean, Nx.tensor([[2.0, 5.0]]))
    end

    test "keeps the true mean of a constant feature, not just its scale" do
      # sklearn.preprocessing.StandardScaler only clamps scale_ to 1 for a
      # constant feature; mean_ is always the real column mean. Zeroing the
      # mean instead (the bug) is observationally different only once you
      # transform data the scaler wasn't fit on, which is the case the two
      # tests below exercise.
      data = Nx.tensor([[1.0, -3.0, 7.0], [2.0, -3.0, 7.0], [3.0, -3.0, 7.0]])

      scaler = StandardScaler.fit(data, axes: [0])

      assert_all_close(scaler.mean, Nx.tensor([[2.0, -3.0, 7.0]]))
      assert_all_close(scaler.standard_deviation, Nx.tensor([[0.8164966, 0.0, 0.0]]))
    end
  end

  describe "transform/2 on data the scaler was not fit on" do
    test "centers a constant feature by its fitted mean instead of passing the raw value through" do
      # Reference: sklearn.preprocessing.StandardScaler().fit([[1,5],[2,5],[3,5]])
      # .transform([[10, 7]]) == [[9.79795897, 2.]]. The second column was
      # constant (5.0) during fit; a value of 7.0 in new data must come out
      # as 7.0 - 5.0 = 2.0, scaled by 1.0 (the clamped scale). Passing 7.0
      # through unscaled would be silent data leakage: the feature stops
      # being centered for exactly the samples that differ from training.
      train = Nx.tensor([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
      scaler = StandardScaler.fit(train, axes: [0])

      new_data = Nx.tensor([[10.0, 7.0]])

      assert_all_close(
        StandardScaler.transform(scaler, new_data),
        Nx.tensor([[9.797958, 2.0]])
      )
    end

    test "matches sklearn for a negative constant feature transformed on new data" do
      # Reference: sklearn StandardScaler().fit([[-2],[-2],[-2]]).transform([[0]]) == [[2.]]
      train = Nx.tensor([[-2.0], [-2.0], [-2.0]])
      scaler = StandardScaler.fit(train, axes: [0])

      assert_all_close(StandardScaler.transform(scaler, Nx.tensor([[0.0]])), Nx.tensor([[2.0]]))
    end
  end
end
