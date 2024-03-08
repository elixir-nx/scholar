defmodule Scholar.Manifold.TrimapTest do
  use Scholar.Case, async: true
  alias Scholar.Manifold.Trimap
  doctest Trimap

  describe "transform" do
    test "non default num_inliers and num_outliers" do
      x = Nx.iota({5, 6})
      key = Nx.Random.key(42)
      res = Trimap.embed(x, num_components: 2, key: key, num_inliers: 3, num_outliers: 1)

      expected =
        Nx.tensor([
          [1.051647424697876, 1.0616474151611328],
          [1.1116474866867065, 1.1216474771499634],
          [1.1716474294662476, 1.1816474199295044],
          [1.2316474914550781, 1.241647481918335],
          [1.2916474342346191, 1.301647424697876]
        ])

      assert_all_close(res, expected)
    end

    test "non default num_random, weight_temp, and learning_rate" do
      x = Nx.iota({5, 6})
      key = Nx.Random.key(42)

      res =
        Trimap.embed(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1,
          num_random: 5,
          weight_temp: 0.1,
          learning_rate: 0.3
        )

      expected =
        Nx.tensor([
          [1.0402824878692627, 1.050282597541809],
          [1.1002825498580933, 1.11028254032135],
          [1.1602826118469238, 1.1702826023101807],
          [1.2202825546264648, 1.2302825450897217],
          [1.2802826166152954, 1.2902826070785522]
        ])

      assert_all_close(res, expected)
    end

    test "non default num_iters and init_embedding_type" do
      x = Nx.iota({5, 6})
      key = Nx.Random.key(42)

      res =
        Trimap.embed(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1,
          num_iters: 100,
          init_embedding_type: 1
        )

      expected =
        Nx.tensor([
          [1.0347665548324585, 1.0537630319595337],
          [1.04896879196167, 1.055889368057251],
          [1.0539816617965698, 1.0542446374893188],
          [1.0503169298171997, 1.0571181774139404],
          [1.0548146963119507, 1.0343143939971924]
        ])

      assert_all_close(res, expected)
    end

    test "passed precomputed triplets and weights" do
      x = Nx.iota({5, 6})
      key = Nx.Random.key(42)
      triplets = Nx.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]])
      weights = Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

      res =
        Trimap.embed(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1,
          triplets: triplets,
          weights: weights
        )

      expected =
        Nx.tensor([
          [1.249335765838623, 0.9553965330123901],
          [1.1023664474487305, 0.9553965330123901],
          [0.9553965330123901, 0.9553965330123901],
          [0.8084271550178528, 0.9553965330123901],
          [0.6614577770233154, 0.9553965330123901]
        ])

      assert_all_close(res, expected)
    end

    test "passed initial_embedding" do
      x = Nx.iota({5, 6})
      key = Nx.Random.key(42)

      init_embeddings =
        Nx.tensor([
          [1.0, 1.0],
          [2.0, 2.0],
          [3.0, 3.0],
          [4.0, 4.0],
          [5.0, 5.0]
        ])

      res =
        Trimap.embed(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1,
          init_embeddings: init_embeddings
        )

      expected =
        Nx.tensor([
          [2.0570054054260254, 2.0570054054260254],
          [3.057004928588867, 3.057004928588867],
          [4.057004451751709, 4.057004451751709],
          [5.057004928588867, 5.057004928588867],
          [6.057004928588867, 6.057004928588867]
        ])

      assert_all_close(res, expected)
    end

    test "metric set to manhattan" do
      x = Nx.iota({5, 6})
      key = Nx.Random.key(42)

      res =
        Trimap.embed(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1,
          metric: :manhattan
        )

      expected =
        Nx.tensor([
          [1.0540052652359009, 1.0640052556991577],
          [1.114005208015442, 1.1240053176879883],
          [1.1740052700042725, 1.1840052604675293],
          [1.2340052127838135, 1.2440052032470703],
          [1.294005274772644, 1.3040052652359009]
        ])

      assert_all_close(res, expected)
    end
  end

  describe "errors" do
    test "invalid num_inliers" do
      x = Nx.iota({2, 6})
      key = Nx.Random.key(42)

      assert_raise ArgumentError,
                   "Number of points must be greater than 2",
                   fn ->
                     Scholar.Manifold.Trimap.embed(x,
                       num_components: 2,
                       key: key,
                       num_inliers: 10,
                       num_outliers: 1
                     )
                   end
    end

    test "triplets and weights with wrong sizes" do
      x = Nx.iota({5, 6})
      key = Nx.Random.key(42)
      triplets = Nx.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]])
      weights = Nx.broadcast(1.0, Nx.shape(triplets))

      assert_raise ArgumentError,
                   "Triplets and weights must be either not initialized or have the same
      size of axis zero and rank of triplets must be 2 and rank of weights must be 1",
                   fn ->
                     Trimap.embed(x,
                       num_components: 2,
                       key: key,
                       num_inliers: 3,
                       num_outliers: 1,
                       triplets: triplets,
                       weights: weights
                     )
                   end
    end
  end
end
