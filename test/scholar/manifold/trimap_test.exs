defmodule Scholar.Manifold.TrimapTest do
  use Scholar.Case, async: true
  alias Scholar.Manifold.Trimap
  doctest Trimap

  describe "transform" do
    test "non default num_inliers and num_outliers" do
      x = Nx.iota({10, 6})
      key = Nx.Random.key(42)

      res =
        Trimap.transform(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1
        )

      expected =
        Nx.tensor([
          [113.99239349365234, 164.60028076171875],
          [111.32695007324219, 164.60028076171875],
          [107.72736358642578, 164.60028076171875],
          [94.22712707519531, 164.60028076171875],
          [77.70183563232422, 164.60028076171875],
          [73.04618835449219, 164.60028076171875],
          [71.35726165771484, 164.60028076171875],
          [61.91230773925781, 164.60028076171875],
          [58.640655517578125, 164.60028076171875],
          [56.583343505859375, 164.60028076171875]
        ])

      assert_all_close(res, expected, atol: 1.0e-3, rtol: 1.0e-3)
    end

    test "non default num_random, weight_temp, and learning_rate" do
      x = Nx.iota({10, 6})
      key = Nx.Random.key(42)

      res =
        Trimap.transform(x,
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
          [112.58768463134766, 164.60028076171875],
          [108.42720794677734, 164.60028076171875],
          [105.17445373535156, 164.60028076171875],
          [94.98436737060547, 164.60028076171875],
          [79.27961730957031, 164.60028076171875],
          [70.53276824951172, 164.60028076171875],
          [65.88448333740234, 164.60028076171875],
          [55.379486083984375, 164.60028076171875],
          [50.87002182006836, 164.60028076171875],
          [49.01177215576172, 164.60028076171875]
        ])

      assert_all_close(res, expected, atol: 1.0e-3, rtol: 1.0e-3)
    end

    test "non default num_iters and init_embedding_type" do
      x = Nx.iota({10, 6})
      key = Nx.Random.key(42)

      res =
        Trimap.transform(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1,
          num_iters: 100,
          init_embedding_type: 1
        )

      expected =
        Nx.tensor([
          [20.231670379638672, 20.449552536010742],
          [19.281051635742188, 19.500879287719727],
          [18.06662368774414, 18.24373435974121],
          [13.929012298583984, 14.01369857788086],
          [9.131621360778809, 9.092915534973145],
          [7.396491050720215, 7.3155999183654785],
          [6.82077169418335, 6.664179801940918],
          [3.6580913066864014, 3.518498182296753],
          [2.479952096939087, 2.3532018661499023],
          [1.4492647647857666, 1.4677170515060425]
        ])

      assert_all_close(res, expected, atol: 1.0e-1, rtol: 1.0e-1)
    end

    test "passed precomputed triplets and weights" do
      x = Nx.iota({10, 6})
      key = Nx.Random.key(42)
      triplets = Nx.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]])
      weights = Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

      res =
        Trimap.transform(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1,
          triplets: triplets,
          weights: weights
        )

      expected =
        Nx.tensor([
          [93.04278564453125, 164.60028076171875],
          [92.46737670898438, 164.60028076171875],
          [82.51052856445312, 164.60028076171875],
          [20.507057189941406, 164.60028076171875],
          [3.474262237548828, 164.60028076171875],
          [164.52679443359375, 164.60028076171875],
          [164.37982177734375, 164.60028076171875],
          [164.23284912109375, 164.60028076171875],
          [164.0858917236328, 164.60028076171875],
          [163.9389190673828, 164.60028076171875]
        ])

      assert_all_close(res, expected, atol: 1.0e-3, rtol: 1.0e-3)
    end

    test "passed initial_embedding" do
      x = Nx.iota({10, 6})
      key = Nx.Random.key(42)

      init_embeddings =
        Nx.tensor([
          [1.0, 1.0],
          [2.0, 2.0],
          [3.0, 3.0],
          [4.0, 4.0],
          [5.0, 5.0],
          [6.0, 6.0],
          [7.0, 7.0],
          [8.0, 8.0],
          [9.0, 9.0],
          [10.0, 10.0]
        ])

      res =
        Trimap.transform(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1,
          init_embeddings: init_embeddings
        )

      expected =
        Nx.tensor([
          [55.56947326660156, 55.56947326660156],
          [58.321434020996094, 58.321434020996094],
          [60.73122787475586, 60.73122787475586],
          [73.6160888671875, 73.6160888671875],
          [88.54448699951172, 88.54448699951172],
          [92.61587524414062, 92.61587524414062],
          [93.69548034667969, 93.69548034667969],
          [103.44023132324219, 103.44023132324219],
          [105.14888000488281, 105.14888000488281],
          [107.56280517578125, 107.56280517578125]
        ])

      assert_all_close(res, expected, atol: 1.0e-3, rtol: 1.0e-3)
    end

    test "metric set to euclidean" do
      x = Nx.iota({10, 6})
      key = Nx.Random.key(42)

      res =
        Trimap.transform(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1,
          knn_algorithm: :brute
        )

      expected =
        Nx.tensor([
          [113.99239349365234, 164.60028076171875],
          [111.32695007324219, 164.60028076171875],
          [107.72736358642578, 164.60028076171875],
          [94.22712707519531, 164.60028076171875],
          [77.70183563232422, 164.60028076171875],
          [73.04618835449219, 164.60028076171875],
          [71.35726165771484, 164.60028076171875],
          [61.91230773925781, 164.60028076171875],
          [58.640655517578125, 164.60028076171875],
          [56.583343505859375, 164.60028076171875]
        ])

      assert_all_close(res, expected, atol: 1.0e-3, rtol: 1.0e-3)
    end
  end

  describe "errors" do
    test "invalid num_inliers" do
      x = Nx.iota({2, 6})
      key = Nx.Random.key(42)

      assert_raise ArgumentError,
                   "Number of points must be greater than 2",
                   fn ->
                     Scholar.Manifold.Trimap.transform(x,
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
                     Trimap.transform(x,
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
