defmodule Scholar.Cluster.AffinityPropagationTest do
  use Scholar.Case, async: true
  alias Scholar.Cluster.AffinityPropagation
  doctest AffinityPropagation

  @seed 42
  @x Nx.tensor([
       [16, 2, 17],
       [17, 3, 9],
       [9, 16, 15],
       [13, 8, 8],
       [3, 5, 15],
       [19, 11, 9],
       [8, 15, 2],
       [16, 2, 2],
       [10, 10, 0],
       [8, 7, 5],
       [4, 11, 8],
       [11, 17, 7],
       [16, 4, 2],
       [13, 9, 7],
       [18, 16, 12],
       [6, 8, 6],
       [18, 13, 1],
       [2, 2, 2],
       [0, 1, 18],
       [12, 16, 18],
       [3, 14, 5],
       [2, 16, 13],
       [6, 6, 13],
       [16, 3, 5],
       [0, 16, 5],
       [4, 18, 5],
       [5, 8, 0],
       [1, 5, 15],
       [10, 0, 14],
       [13, 8, 14],
       [19, 2, 9],
       [17, 17, 0],
       [19, 14, 19],
       [9, 19, 10],
       [11, 4, 12],
       [3, 16, 19],
       [17, 3, 6],
       [9, 16, 10],
       [5, 17, 3],
       [3, 15, 17]
     ])
  @x_test Nx.tensor([
            [12, 11, 15],
            [6, 3, 3],
            [8, 16, 16],
            [12, 2, 17],
            [11, 3, 17],
            [15, 1, 14],
            [0, 6, 7],
            [7, 9, 3],
            [13, 3, 16],
            [11, 2, 2]
          ])

  test "fit and compute_values" do
    vals = AffinityPropagation.fit(@x, seed: @seed)

    model = AffinityPropagation.compute_values(vals)

    assert model.labels ==
             Nx.tensor([
               5,
               6,
               0,
               2,
               1,
               3,
               7,
               6,
               2,
               2,
               2,
               7,
               6,
               2,
               3,
               2,
               4,
               2,
               1,
               0,
               7,
               0,
               1,
               6,
               7,
               7,
               2,
               1,
               5,
               5,
               6,
               4,
               3,
               0,
               5,
               0,
               6,
               0,
               7,
               0
             ])

    assert model.cluster_centers ==
             Nx.tensor([
               [9, 16, 15],
               [3, 5, 15],
               [8, 7, 5],
               [18, 16, 12],
               [18, 13, 1],
               [11, 4, 12],
               [17, 3, 6],
               [5, 17, 3]
             ])

    assert model.cluster_centers_indices == Nx.tensor([2, 4, 9, 14, 16, 34, 36, 38])
  end

  test "predict" do
    vals = AffinityPropagation.fit(@x, seed: @seed)
    model = AffinityPropagation.compute_values(vals)
    preds = AffinityPropagation.predict(model, @x_test)
    assert preds == Nx.tensor([0, 2, 0, 5, 5, 5, 2, 2, 5, 2])
  end
end
