defmodule Scholar.Metrics.DistanceTest do
  use Scholar.Case
  alias Scholar.Metrics.Distance
  doctest Distance

  @x Nx.tensor([
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

  @y Nx.tensor([
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

  test "euclidean matches scipy" do
    assert_all_close(Distance.euclidean(@x, @y), Nx.tensor(3.388213202573845))
  end

  test "squared euclidean matches scipy" do
    assert_all_close(Distance.squared_euclidean(@x, @y), Nx.tensor(11.479988706095714))
  end

  test "manhattan matches scipy" do
    assert_all_close(Distance.manhattan(@x, @y), Nx.tensor(8.694822449237108))
  end

  test "chebyshev matches scipy" do
    assert_all_close(Distance.chebyshev(@x, @y), Nx.tensor(2.043651282787323))
  end

  test "minkowski matches scipy" do
    assert_all_close(Distance.minkowski(@x, @y), Nx.tensor(3.388213202573845))
  end

  test "cosine matches scipy" do
    assert_all_close(Distance.cosine(@x, @y), Nx.tensor(0.7650632810164779))
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
end
