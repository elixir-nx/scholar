defmodule Scholar.Cluster.OPTICSTest do
  use Scholar.Case, async: true

  alias Scholar.Cluster.OPTICS
  doctest OPTICS

  # Expected values from scikit-learn 1.6.1 on this data.
  defp x, do: Nx.tensor([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]], type: :f64)

  test "fit/2 returns the reachability graph" do
    model = OPTICS.fit(x(), eps: 4.5, min_samples: 2)

    assert model.ordering == Nx.tensor([0, 1, 2, 5, 3, 4], type: :s32)
    assert model.predecessor == Nx.tensor([-1, 0, 1, 5, 3, 2], type: :s32)

    assert_all_close(
      model.core_distances,
      Nx.tensor([3.16227766, 1.41421356, 1.41421356, 1.0, 1.0, 4.12310563], type: :f64)
    )

    assert_all_close(
      model.reachability,
      Nx.tensor([:infinity, 3.16227766, 1.41421356, 4.12310563, 1.0, 5.0], type: :f64)
    )
  end

  test "fit/2 caps the reachability graph at max_eps" do
    model = OPTICS.fit(x(), max_eps: 2, min_samples: 2)

    assert model.labels == Nx.tensor([-1, 0, 0, 1, 1, -1], type: :s32)

    assert_all_close(
      model.core_distances,
      Nx.tensor([:infinity, 1.41421356, 1.41421356, 1.0, 1.0, :infinity], type: :f64)
    )

    assert_all_close(
      model.reachability,
      Nx.tensor([:infinity, :infinity, 1.41421356, :infinity, 1.0, :infinity], type: :f64)
    )
  end
end
