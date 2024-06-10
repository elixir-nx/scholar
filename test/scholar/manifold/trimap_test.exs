defmodule Scholar.Manifold.TrimapTest do
  use Scholar.Case, async: true
  alias Scholar.Manifold.Trimap
  doctest Trimap

  describe "transform" do
    test "non default num_inliers and num_outliers" do
      x = Nx.iota({5, 6})
      key = Nx.Random.key(42)

      res =
        Trimap.embed(x,
          num_components: 2,
          key: key,
          num_inliers: 3,
          num_outliers: 1,
          algorithm: :nndescent
        )

      expected =
        Nx.tensor([
          [3.3822429180145264, 3.392242908477783],
          [3.4422430992126465, 3.4522430896759033],
          [3.5022432804107666, 3.5122432708740234],
          [3.5622432231903076, 3.5722432136535645],
          [3.6222431659698486, 3.6322431564331055]
        ])

      assert_all_close(res, expected, atol: 1.0e-3, rtol: 1.0e-3)
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
          learning_rate: 0.3,
          algorithm: :nndescent
        )

      expected =
        Nx.tensor([
          [3.352140426635742, 3.362140417098999],
          [3.412140369415283, 3.42214035987854],
          [3.472140312194824, 3.482140302658081],
          [3.5321402549743652, 3.542140245437622],
          [3.5921404361724854, 3.602140426635742]
        ])

      assert_all_close(res, expected, atol: 1.0e-3, rtol: 1.0e-3)
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
          init_embedding_type: 1,
          algorithm: :nndescent
        )

      expected =
        Nx.tensor([
          [1.4574551582336426, 1.443753719329834],
          [1.4331351518630981, 1.4537053108215332],
          [1.4543260335922241, 1.4485278129577637],
          [1.4427212476730347, 1.4643783569335938],
          [1.449319839477539, 1.455613374710083]
        ])

      assert_all_close(res, expected, atol: 1.0e-1, rtol: 1.0e-1)
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
          weights: weights,
          algorithm: :nndescent
        )

      expected =
        Nx.tensor([
          [2.822676420211792, 3.116623878479004],
          [2.9696502685546875, 3.116623878479004],
          [3.116623878479004, 3.116623878479004],
          [3.263594388961792, 3.116623878479004],
          [3.4105637073516846, 3.116623878479004]
        ])

      assert_all_close(res, expected, atol: 1.0e-3, rtol: 1.0e-3)
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
          init_embeddings: init_embeddings,
          algorithm: :nndescent
        )

      expected =
        Nx.tensor([
          [4.396871089935303, 4.396871089935303],
          [5.396846771240234, 5.396846771240234],
          [6.396846771240234, 6.396846771240234],
          [7.396847724914551, 7.396847724914551],
          [8.39684772491455, 8.39684772491455]
        ])

      assert_all_close(res, expected, atol: 1.0e-3, rtol: 1.0e-3)
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
          metric: :manhattan,
          algorithm: :nndescent
        )

      expected =
        Nx.tensor([
          [3.3887996673583984, 3.3987996578216553],
          [3.4487998485565186, 3.4587998390197754],
          [3.5087997913360596, 3.5187997817993164],
          [3.5687997341156006, 3.5787997245788574],
          [3.6287999153137207, 3.6387999057769775]
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
                       weights: weights,
                       algorithm: :nndescent
                     )
                   end
    end
  end
end
