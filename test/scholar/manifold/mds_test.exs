defmodule Scholar.Manifold.MDSTest do
  use Scholar.Case, async: true
  alias Scholar.Manifold.MDS
  doctest MDS

  def x() do
    Nx.iota({10, 50})
  end

  def key() do
    Nx.Random.key(42)
  end

  test "all default" do
    model = MDS.fit(x(), key: key())

    assert_all_close(
      model.embedding,
      Nx.tensor([
        [-1200.2181396484375, -1042.11083984375],
        [-985.0137939453125, -750.6790771484375],
        [-706.14013671875, -532.040771484375],
        [-402.91387939453125, -344.8670959472656],
        [-163.916015625, -77.55931091308594],
        [137.63134765625, 111.43733215332031],
        [450.9678649902344, 284.375],
        [712.8345947265625, 524.7731323242188],
        [935.5824584960938, 807.938720703125],
        [1221.1859130859375, 1018.7330932617188]
      ])
    )

    assert_all_close(model.stress, 390.99090576171875)
    assert_all_close(model.n_iter, Nx.tensor(93))
  end

  test "non-default num_components" do
    model = MDS.fit(x(), num_components: 5, key: key())

    assert_all_close(
      model.embedding,
      Nx.tensor([
        [
          -753.6793823242188,
          -1215.4837646484375,
          -604.7247314453125,
          136.4171905517578,
          306.40069580078125
        ],
        [
          -536.5817260742188,
          -982.4620971679688,
          -457.0782165527344,
          113.92396545410156,
          232.4468994140625
        ],
        [
          -368.63018798828125,
          -696.4574584960938,
          -344.57952880859375,
          120.76351165771484,
          164.8445281982422
        ],
        [
          -216.7689666748047,
          -406.25250244140625,
          -248.1519317626953,
          85.04608154296875,
          65.24085235595703
        ],
        [
          -52.247528076171875,
          -147.1763916015625,
          -101.35457611083984,
          38.1723747253418,
          -30.632429122924805
        ],
        [
          112.39735412597656,
          137.04685974121094,
          19.412824630737305,
          -15.030402183532715,
          -70.50560760498047
        ],
        [
          270.4787902832031,
          401.2673034667969,
          195.0669403076172,
          -36.55331039428711,
          -105.96172332763672
        ],
        [
          381.5384216308594,
          700.9933471679688,
          348.5281982421875,
          -61.6308708190918,
          -142.2161407470703
        ],
        [
          508.359130859375,
          980.1181030273438,
          507.714111328125,
          -153.78958129882812,
          -164.67311096191406
        ],
        [
          655.1340942382812,
          1228.4066162109375,
          685.1668701171875,
          -227.31895446777344,
          -254.94395446777344
        ]
      ])
    )

    assert_all_close(model.stress, 641.52490234375)
    assert_all_close(model.n_iter, Nx.tensor(130))
  end

  test "non-default metric" do
    model = MDS.fit(x(), metric: false, key: key())

    assert_all_close(
      model.embedding,
      Nx.tensor([
        [0.4611709713935852, -0.2790529131889343],
        [0.10750522464513779, 0.3869015574455261],
        [0.10845339298248291, -0.619588315486908],
        [-0.3274216949939728, -0.2036580592393875],
        [0.432122141122818, 0.4288368821144104],
        [-0.2664470970630646, 0.1712798774242401],
        [-0.46502357721328735, 0.015750018879771233],
        [0.35657963156700134, 0.028018075972795486],
        [-0.11095760017633438, -0.3872125744819641],
        [-0.20736312866210938, 0.41101184487342834]
      ])
    )

    assert_all_close(model.stress, 1.2879878282546997)
    assert_all_close(model.n_iter, Nx.tensor(18))
  end

  test "option normalized_stress with metric set to false" do
    model =
      MDS.fit(x(), metric: false, key: key(), normalized_stress: true)

    assert_all_close(
      model.embedding,
      Nx.tensor([
        [-0.5107499957084656, -0.5828369855880737],
        [-0.008806264027953148, -0.4549526870250702],
        [-0.5534653663635254, -0.02513509802520275],
        [0.11427811533212662, 0.17350295186042786],
        [0.45669451355934143, 0.20050597190856934],
        [0.010616336017847061, -0.09705149382352829],
        [-0.27859434485435486, 0.3822994530200958],
        [0.353694885969162, -0.17320780456066132],
        [0.49716615676879883, -0.10724353790283203],
        [-0.12109922617673874, 0.4835425913333893]
      ])
    )

    assert_all_close(model.stress, 0.24878354370594025)
    assert_all_close(model.n_iter, Nx.tensor(8))
  end

  test "epsilon set to a smaller then default value" do
    model = MDS.fit(x(), key: key(), eps: 1.0e-4)

    assert_all_close(
      model.embedding,
      Nx.tensor([
        [-1210.0882568359375, -1031.7977294921875],
        [-975.5465087890625, -762.0400390625],
        [-702.4406127929688, -536.8583984375],
        [-407.59564208984375, -339.2669372558594],
        [-155.48812866210938, -88.38337707519531],
        [139.16014099121094, 109.21498107910156],
        [439.32861328125, 299.55438232421875],
        [705.1881713867188, 533.9835205078125],
        [944.5883178710938, 798.3893432617188],
        [1222.8939208984375, 1017.2042846679688]
      ])
    )

    # as expected smaller value of stress (loss) and bigger number of iterations that all default
    assert_all_close(model.stress, 86.7530288696289)
    assert_all_close(model.n_iter, Nx.tensor(197))
  end

  test "smaller max_iter value (100)" do
    model = MDS.fit(x(), key: key(), eps: 1.0e-4, max_iter: 100)

    assert_all_close(
      model.embedding,
      Nx.tensor([
        [-1201.354736328125, -1040.9530029296875],
        [-983.8963012695312, -752.0218505859375],
        [-705.769775390625, -532.5292358398438],
        [-403.48602294921875, -344.1869201660156],
        [-162.94749450683594, -78.80620574951172],
        [137.8325958251953, 111.14128875732422],
        [449.6497497558594, 286.1120300292969],
        [711.979736328125, 525.80078125],
        [936.6356811523438, 806.8446655273438],
        [1221.3565673828125, 1018.59814453125]
      ])
    )

    # same params as in previous test, but smaller number of iterations, cupped on 100
    assert_all_close(model.stress, 337.1789245605469)
    assert_all_close(model.n_iter, Nx.tensor(100))
  end
end
