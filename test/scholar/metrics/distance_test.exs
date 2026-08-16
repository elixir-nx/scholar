defmodule Scholar.Metrics.DistanceTest do
  use Scholar.Case, async: true
  alias Scholar.Metrics.Distance
  doctest Distance

  defp x do
    Nx.tensor([
      -0.165435329079628,
      -1.0515050888061523,
      1.8801462650299072,
      0.2381746470928192,
      0.6978269219398499,
      0.025831177830696106,
      0.11569870263338089,
      -0.6905220150947571,
      -0.9335482120513916,
      -0.025539811700582504
    ])
  end

  defp y do
    Nx.tensor([
      0.5898482203483582,
      -0.5769372582435608,
      1.43277108669281,
      -0.024414867162704468,
      -1.3458243608474731,
      1.669877052307129,
      0.6263275742530823,
      0.8154261708259583,
      0.06888432800769806,
      0.022759810090065002
    ])
  end

  test "euclidean matches scipy" do
    assert_all_close(Distance.euclidean(x(), y()), Nx.tensor(3.388213202573845))
  end

  test "squared euclidean matches scipy" do
    assert_all_close(Distance.squared_euclidean(x(), y()), Nx.tensor(11.479988706095714))
  end

  test "manhattan matches scipy" do
    assert_all_close(Distance.manhattan(x(), y()), Nx.tensor(8.694822449237108))
  end

  test "chebyshev matches scipy" do
    assert_all_close(Distance.chebyshev(x(), y()), Nx.tensor(2.043651282787323))
  end

  test "minkowski matches scipy" do
    assert_all_close(Distance.minkowski(x(), y()), Nx.tensor(3.388213202573845))
  end

  test "minkowski with p set to :infinity matches chebyshev" do
    assert_all_close(Distance.minkowski(x(), y(), p: :infinity), Nx.tensor(2.043651282787323))
  end

  test "cosine matches scipy" do
    assert_all_close(Distance.cosine(x(), y()), Nx.tensor(0.7650632810164779))
  end

  test "hamming matches scipy" do
    assert Distance.hamming(Nx.tensor([1, 0, 0]), Nx.tensor([0, 1, 0])) ==
             Nx.tensor(0.6666666865348816)

    assert Distance.hamming(Nx.tensor([1, 0, 0]), Nx.tensor([1, 1, 0])) ==
             Nx.tensor(0.3333333432674408)

    assert Distance.hamming(Nx.tensor([1, 0, 0]), Nx.tensor([2, 0, 0])) ==
             Nx.tensor(0.3333333432674408)

    assert Distance.hamming(Nx.tensor([1, 0, 0]), Nx.tensor([3, 0, 0])) ==
             Nx.tensor(0.3333333432674408)
  end

  describe "pairwise_squared_euclidean precision" do
    test "large-magnitude f32 coordinates do not catastrophically cancel" do
      x = Nx.tensor([[100_000.0, 100_000.0]], type: :f32)
      y = Nx.tensor([[100_000.0, 100_000.1]], type: :f32)

      assert_all_close(Distance.pairwise_squared_euclidean(x, y), Nx.tensor([[0.01]]),
        atol: 1.0e-3
      )
    end

    test "very large f64 coordinates do not overflow" do
      x = Nx.tensor([[1.0e200, 0.0]], type: :f64)
      y = Nx.tensor([[1.0e200, 1.0]], type: :f64)

      assert_all_close(Distance.pairwise_squared_euclidean(x, y), Nx.tensor([[1.0]]))
    end

    test "a non-finite sentinel row does not poison the other rows" do
      # A row that is entirely :infinity (as Scholar.Cluster.AffinityPropagation
      # uses as a placeholder for not-yet-selected exemplars) can only ever
      # produce :infinity or NaN against it, never a small/wrong finite value
      # that could win an argmin. Callers already normalize NaN to :infinity
      # (see AffinityPropagation.predict/2), so that normalized result must
      # always be :infinity.
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      y = Nx.tensor([[:infinity, :infinity], [1.0, 2.0]])

      result = Distance.pairwise_squared_euclidean(x, y)
      normalized = Nx.select(Nx.is_nan(result), Nx.Constants.infinity(Nx.type(result)), result)

      assert_all_close(result[[.., 1]], Nx.tensor([0.0, 8.0]))
      assert Nx.to_number(Nx.all(Nx.is_infinity(normalized[[.., 0]]))) == 1
    end

    test "a non-finite sentinel row does not poison the other rows (single-argument form)" do
      x = Nx.tensor([[:infinity, :infinity], [1.0, 2.0], [3.0, 4.0]])

      result = Distance.pairwise_squared_euclidean(x)
      normalized = Nx.select(Nx.is_nan(result), Nx.Constants.infinity(Nx.type(result)), result)

      assert_all_close(result[[1..2, 1..2]], Nx.tensor([[0.0, 8.0], [8.0, 0.0]]))
      assert Nx.to_number(Nx.all(Nx.is_infinity(normalized[[0, 1..2]]))) == 1
    end

    test "integer input stays exact and integer-typed" do
      x = Nx.tensor([[1, 2, 5], [3, 4, 3]])
      y = Nx.tensor([[8, 3, 1], [2, 5, 2]])

      result = Distance.pairwise_squared_euclidean(x, y)

      assert Nx.type(result) == {:s, 32}
      assert result == Nx.tensor([[66, 19], [30, 3]])
    end

    test "works with jit_apply" do
      x = Nx.tensor([[100_000.0, 100_000.0]], type: :f32)
      y = Nx.tensor([[100_000.0, 100_000.1]], type: :f32)

      jit_result = Nx.Defn.jit_apply(&Distance.pairwise_squared_euclidean/2, [x, y])
      assert_all_close(jit_result, Distance.pairwise_squared_euclidean(x, y))
    end
  end
end
