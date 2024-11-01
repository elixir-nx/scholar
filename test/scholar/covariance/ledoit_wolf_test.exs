defmodule Scholar.Covariance.LedoitWolfTest do
  use Scholar.Case, async: true
  alias Scholar.Covariance.LedoitWolf
  doctest LedoitWolf

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

    model = LedoitWolf.fit(x)

    assert_all_close(
      model.covariance,
      Nx.tensor([
        [1.439786434173584, -0.0, 0.0],
        [-0.0, 1.439786434173584, 0.0],
        [0.0, 0.0, 1.439786434173584]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(model.shrinkage, Nx.tensor(1.0), atol: 1.0e-3)

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

    model = LedoitWolf.fit(x, assume_centered: true)

    assert_all_close(
      model.covariance,
      Nx.tensor([
        [1.852303147315979, 0.0, 0.0],
        [0.0, 1.852303147315979, 0.0],
        [0.0, 0.0, 1.852303147315979]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(model.shrinkage, Nx.tensor(1.0), atol: 1.0e-3)

    assert_all_close(model.location, Nx.tensor([0, 0, 0]), atol: 1.0e-3)
  end

  test "fit test - set :block_size" do
    key = key()

    {x, _new_key} =
      Nx.Random.multivariate_normal(
        key,
        Nx.tensor([0.0, 0.0]),
        Nx.tensor([[2.2, 1.5], [0.7, 1.1]]),
        shape: {50},
        type: :f32
      )

    model = LedoitWolf.fit(x, block_size: 20)

    assert_all_close(
      model.covariance,
      Nx.tensor([
        [1.8378269672393799, 0.27215731143951416],
        [0.27215731143951416, 1.2268550395965576]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(model.shrinkage, Nx.tensor(0.38731059432029724), atol: 1.0e-3)

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

    model = LedoitWolf.fit(x)

    assert_all_close(
      model.covariance,
      Nx.tensor([
        [0.5322133302688599]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(model.shrinkage, Nx.tensor(0.0), atol: 1.0e-3)

    assert_all_close(model.location, Nx.tensor([0.060818854719400406]), atol: 1.0e-3)
  end
end
