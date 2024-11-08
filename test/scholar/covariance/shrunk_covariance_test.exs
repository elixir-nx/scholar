defmodule Scholar.Covariance.ShrunkCovarianceTest do
  use Scholar.Case, async: true
  alias Scholar.Covariance.ShrunkCovariance
  doctest ShrunkCovariance

  defp key do
    Nx.Random.key(1)
  end

  test "fit test - all default options" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(
        key,
        Nx.tensor([0.0, 0.0, 0.0]),
        Nx.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.3, 1.0, 2.2]]),
        shape: {10},
        type: :f32
      )

    model = ShrunkCovariance.fit(x)

    assert_all_close(
      model.covariance,
      Nx.tensor([
        [2.0949244499206543, -0.13400490581989288, 0.5413897037506104],
        [-0.13400490581989288, 1.2940725088119507, 0.0621684193611145],
        [0.5413897037506104, 0.0621684193611145, 0.9303621053695679]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.location,
      Nx.tensor([-1.015519142150879, -0.4495307505130768, 0.06475571542978287]),
      atol: 1.0e-3
    )
  end

  test "fit test - :assume_centered is true" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(
        key,
        Nx.tensor([0.0, 0.0, 0.0]),
        Nx.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.3, 1.0, 2.2]]),
        shape: {10},
        type: :f32
      )

    model = ShrunkCovariance.fit(x, assume_centered: true)

    assert_all_close(
      model.covariance,
      Nx.tensor([
        [3.0643274784088135, 0.27685147523880005, 0.4822050631046295],
        [0.27685147523880005, 1.5171942710876465, 0.03596973791718483],
        [0.4822050631046295, 0.03596973791718483, 0.975387692451477]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(model.location, Nx.tensor(0), atol: 1.0e-3)
  end

  test "fit test - :shrinkage" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(
        key,
        Nx.tensor([0.0, 0.0, 0.0]),
        Nx.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.3, 1.0, 2.2]]),
        shape: {10},
        type: :f32
      )

    model = ShrunkCovariance.fit(x, shrinkage: 0.8)

    assert_all_close(
      model.covariance,
      Nx.tensor([
        [1.5853726863861084, -0.029778867959976196, 0.12030883133411407],
        [-0.029778867959976196, 1.4074056148529053, 0.013815204612910748],
        [0.12030883133411407, 0.013815204612910748, 1.3265810012817383]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.location,
      Nx.tensor([-1.015519142150879, -0.4495307505130768, 0.06475571542978287]),
      atol: 1.0e-3
    )
  end

  test "fit test 2" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(
        key,
        Nx.tensor([0.0, 0.0]),
        Nx.tensor([[2.2, 1.5], [0.7, 1.1]]),
        shape: {50},
        type: :f32
      )

    model = ShrunkCovariance.fit(x)

    assert_all_close(
      model.covariance,
      Nx.tensor([
        [1.9810796976089478, 0.3997809886932373],
        [0.3997809886932373, 1.0836023092269897]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(model.location, Nx.tensor([0.06882287561893463, 0.13750331103801727]),
      atol: 1.0e-3
    )
  end

  test "fit test - 1 dim x" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(key, Nx.tensor([0.0]), Nx.tensor([[0.4]]),
        shape: {15},
        type: :f32
      )

    x = Nx.flatten(x)

    model = ShrunkCovariance.fit(x)

    assert_all_close(
      model.covariance,
      Nx.tensor([
        [0.5322133302688599]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(model.location, Nx.tensor([0.060818854719400406]), atol: 1.0e-3)
  end
end
