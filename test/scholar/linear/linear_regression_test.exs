defmodule Scholar.Linear.LinearRegressionTest do
  use Scholar.Case, async: true

  describe "fit" do
    test "matches sklearn for shapes {1, 1}, {1, 1} and type {:f, 32}" do
      a = Nx.tensor([[0.5666993856430054]])
      b = Nx.tensor([[0.8904717564582825]])
      expected_coeff = Nx.tensor([[0.0]])
      expected_intercept = Nx.tensor([0.89047176])

      %Scholar.Linear.LinearRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        Scholar.Linear.LinearRegression.fit(a, b)

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "matches sklearn for shapes {4, 6}, {4} and type {:f, 32}" do
      a =
        Nx.tensor([
          [
            0.9106853604316711,
            0.2549760937690735,
            0.6961511969566345,
            0.9727275371551514,
            0.21542717516422272,
            0.1134142205119133
          ],
          [
            0.06984847038984299,
            0.5999580025672913,
            0.5002026557922363,
            0.3873320519924164,
            0.4304788112640381,
            0.7481111884117126
          ],
          [
            0.8257377743721008,
            0.3594258427619934,
            0.08661065995693207,
            0.5118331909179688,
            0.38879409432411194,
            0.3640798032283783
          ],
          [
            0.060781270265579224,
            0.054385241121053696,
            0.6188914775848389,
            0.9003549218177795,
            0.7764868140220642,
            0.9584161043167114
          ]
        ])

      b =
        Nx.tensor([0.608439028263092, 0.6562057137489319, 0.9454836249351501, 0.8614323735237122])

      expected_coeff =
        Nx.tensor([
          0.14827239513397217,
          -0.2630932927131653,
          -0.45938295125961304,
          -0.02042902633547783,
          0.2715783417224884,
          0.16200493276119232
        ])

      expected_intercept = Nx.tensor(0.8032849746598018)

      %Scholar.Linear.LinearRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        Scholar.Linear.LinearRegression.fit(a, b)

      assert_all_close(expected_coeff, actual_coeff, rtol: 1.0e-2, atol: 1.0e-2)
      assert_all_close(expected_intercept, actual_intercept, rtol: 1.0e-2, atol: 1.0e-3)
    end

    test "matches sklearn for shapes {6, 6}, {6, 1} and type {:f, 64}" do
      a =
        Nx.tensor([
          [
            0.33747746329154127,
            0.8448716981481403,
            0.8465694977489985,
            0.1926030183919778,
            0.04625514004748599,
            0.7502944398364579
          ],
          [
            0.4446625676763911,
            0.8463476875150429,
            0.39503640704174303,
            0.7910270477615085,
            0.8722376324896636,
            0.6758646358483182
          ],
          [
            0.6154292118141929,
            0.5455230739505744,
            0.9565376231248434,
            0.2790218491103198,
            0.5663205639536116,
            0.29588894254993525
          ],
          [
            0.6873114496145727,
            0.2603452300422152,
            0.5479350062232057,
            0.5267668983186267,
            0.2557562799821602,
            0.4790844622306156
          ],
          [
            0.3298696032797205,
            0.3446971837357009,
            0.2888187784379451,
            0.6165562827943281,
            0.27242014359429534,
            0.0243891670454095
          ],
          [
            0.8073663574129741,
            0.6744673959108053,
            0.24853954732383965,
            0.26991916232511237,
            0.3544102499522487,
            0.8091680144952467
          ]
        ])

      b =
        Nx.tensor([
          [0.7471436930806791],
          [0.17232638941796508],
          [0.2709342130703567],
          [0.8110701184628758],
          [0.936094841986774],
          [0.4667805339798258]
        ])

      expected_coeff =
        Nx.tensor([
          [
            -0.3777002030151436,
            -0.4445957357428203,
            -0.14451413829286042,
            0.31438593891571714,
            -0.9484560114249797,
            0.04914973264178196
          ]
        ])

      expected_intercept = Nx.tensor([1.31901913])

      %Scholar.Linear.LinearRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        Scholar.Linear.LinearRegression.fit(a, b)

      assert_all_close(expected_coeff, actual_coeff, rtol: 1.0e-2, atol: 1.0e-2)
      assert_all_close(expected_intercept, actual_intercept, rtol: 1.0e-2, atol: 1.0e-3)
    end

    test "matches sklearn for shapes {8, 6}, {8, 4} and type {:f, 32}" do
      a =
        Nx.tensor([
          [
            0.4486805200576782,
            0.5008742213249207,
            0.8596201539039612,
            0.1028851643204689,
            0.3482196033000946,
            0.7360767722129822
          ],
          [
            0.5973052978515625,
            0.42625781893730164,
            0.6490350961685181,
            0.9295122623443604,
            0.2793756127357483,
            0.2399832159280777
          ],
          [
            0.49139007925987244,
            0.3047942519187927,
            0.1499510407447815,
            0.17941612005233765,
            0.35138219594955444,
            0.10164441168308258
          ],
          [
            0.12782688438892365,
            0.9228906631469727,
            0.14866666495800018,
            0.6292259097099304,
            0.7426121234893799,
            0.6266968250274658
          ],
          [
            0.8389331698417664,
            0.0007423785864375532,
            0.6050190925598145,
            0.6548615097999573,
            0.27365484833717346,
            0.6509554386138916
          ],
          [
            0.45333459973335266,
            0.2806072533130646,
            0.7201200723648071,
            0.25043684244155884,
            0.8272882103919983,
            0.4810141921043396
          ],
          [
            0.8025938272476196,
            0.4874570965766907,
            0.7029373645782471,
            0.6005574464797974,
            0.3122026026248932,
            0.2309170365333557
          ],
          [
            0.9924272298812866,
            0.4458909332752228,
            0.45357561111450195,
            0.3934052884578705,
            0.703323483467102,
            0.23038771748542786
          ]
        ])

      b =
        Nx.tensor([
          [0.5419812798500061, 0.7586863040924072, 0.0155774662271142, 0.9477185010910034],
          [0.41905564069747925, 0.6679878830909729, 0.06412497162818909, 0.6899121999740601],
          [0.939400315284729, 0.1704750657081604, 0.5241623520851135, 0.027816439047455788],
          [0.4943249821662903, 0.23289655148983002, 0.11643275618553162, 0.008191977627575397],
          [0.45557326078414917, 0.5481954216957092, 0.893762469291687, 0.14807121455669403],
          [0.6307262778282166, 0.9006569981575012, 0.15309521555900574, 0.8714659810066223],
          [0.31813278794288635, 0.1615457981824875, 0.986994206905365, 0.896649956703186],
          [0.000647166685666889, 0.35463055968284607, 0.17690572142601013, 0.6353870630264282]
        ])

      expected_coeff =
        Nx.tensor([
          [
            -1.0994031429290771,
            -0.7150872945785522,
            -0.14029383659362793,
            -0.2482454478740692,
            -0.34398719668388367,
            -0.31880879402160645
          ],
          [
            -0.5908892750740051,
            -0.6351519823074341,
            0.8402842283248901,
            0.1367672085762024,
            0.5045166015625,
            0.038073837757110596
          ],
          [
            0.5530288219451904,
            -0.338871031999588,
            -0.49284127354621887,
            0.08113034814596176,
            -0.608875572681427,
            0.20944960415363312
          ],
          [
            0.0426587350666523,
            0.5576832294464111,
            1.7411061525344849,
            -0.22373059391975403,
            0.31550341844558716,
            -0.6994999647140503
          ]
        ])

      expected_intercept = Nx.tensor([1.9170046, 0.3207544, 0.61264162, -0.42393678])

      %Scholar.Linear.LinearRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        Scholar.Linear.LinearRegression.fit(a, b)

      assert_all_close(expected_coeff, actual_coeff, rtol: 1.0e-1, atol: 1.0e-2)
      assert_all_close(expected_intercept, actual_intercept, rtol: 1.0e-1, atol: 1.0e-2)
    end

    test "matches sklearn for shapes {1, 1}, {1, 1} and type {:f, 32} and sample_weights" do
      a = Nx.tensor([[0.3166404366493225]])
      b = Nx.tensor([[0.6253954172134399]])
      sample_weights = [0.2065236121416092]
      expected_coeff = Nx.tensor([[0.0]])
      expected_intercept = Nx.tensor([0.62539542])

      %Scholar.Linear.LinearRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        Scholar.Linear.LinearRegression.fit(a, b, sample_weights: sample_weights)

      assert_all_close(expected_coeff, actual_coeff)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "matches sklearn for shapes {4, 6}, {4} and type {:f, 32} and sample_weight" do
      a =
        Nx.tensor([
          [
            0.7731901407241821,
            0.5813425779342651,
            0.8365984559059143,
            0.2182593196630478,
            0.06448899209499359,
            0.9420031905174255
          ],
          [
            0.6547101736068726,
            0.05023770406842232,
            0.657528281211853,
            0.24924135208129883,
            0.8238568902015686,
            0.11182288080453873
          ],
          [
            0.7693489193916321,
            0.6696648001670837,
            0.6877049803733826,
            0.08740159869194031,
            0.6053816676139832,
            0.5419610142707825
          ],
          [
            0.03419172018766403,
            0.8298202753067017,
            0.6097439527511597,
            0.0184243805706501,
            0.5578944087028503,
            0.9986271858215332
          ]
        ])

      b =
        Nx.tensor([
          0.38682249188423157,
          0.8040792346000671,
          0.8069542646408081,
          0.3620224595069885
        ])

      sample_weights = [
        0.8669093251228333,
        0.10421276837587357,
        0.996828556060791,
        0.29747673869132996
      ]

      expected_coeff =
        Nx.tensor([0.42989581, 0.29024197, -0.07561158, -0.15607637, 0.39304042, -0.37964429])

      expected_intercept = Nx.tensor(0.3153022844013592)

      %Scholar.Linear.LinearRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        Scholar.Linear.LinearRegression.fit(a, b, sample_weights: sample_weights)

      assert_all_close(expected_coeff, actual_coeff, rtol: 1.0e-1, atol: 1.0e-2)
      assert_all_close(expected_intercept, actual_intercept, rtol: 1.0e-3, atol: 1.0e-2)
    end

    test "matches sklearn for shapes {6, 6}, {6, 1} and type {:f, 64} and sample_weight" do
      a =
        Nx.tensor([
          [
            0.34724389243044596,
            0.07108068156211422,
            0.7976891790264992,
            0.7720415894582006,
            0.9931094599499067,
            0.042191702350034554
          ],
          [
            0.8976550768631645,
            0.27821791769675597,
            0.782111252555358,
            0.07054992729514142,
            0.4147036460728568,
            0.3364791695200583
          ],
          [
            0.2420664763199637,
            0.7610433293120353,
            0.9563727610512863,
            0.6783483347933696,
            0.8653053562992926,
            0.358786243983936
          ],
          [
            0.8590635244018447,
            0.4629443449654602,
            0.7127744295313074,
            0.5560331918259482,
            0.08071683729330104,
            0.024406512158064664
          ],
          [
            0.2058524918820357,
            0.031190848801941007,
            0.7219352939086864,
            0.24717854915189252,
            0.7850315358774881,
            0.568463342839088
          ],
          [
            0.6093294137942888,
            0.2895127677108337,
            0.688868517174993,
            0.9101196747532645,
            0.01929359320504298,
            0.6521018951397942
          ]
        ])

      b =
        Nx.tensor([
          [0.4987439089281194],
          [0.18462047947658167],
          [0.7948645467622648],
          [0.12639533359219102],
          [0.8324762751780334],
          [0.04369688910637193]
        ])

      sample_weights = [
        0.9732323515038417,
        0.2329840671446376,
        0.20240177758428113,
        0.1389157480770663,
        0.6516564408412683,
        0.08165184634168055
      ]

      expected_coeff =
        Nx.tensor([[-1.252728, 0.33221864, -0.23523702, -0.53585187, 0.00157968, -0.24489391]])

      expected_intercept = Nx.tensor([1.52024138])

      %Scholar.Linear.LinearRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        Scholar.Linear.LinearRegression.fit(a, b, sample_weights: sample_weights)

      assert_all_close(expected_coeff, actual_coeff, rtol: 1.0e-2, atol: 1.0e-2)
      assert_all_close(expected_intercept, actual_intercept, rtol: 1.0e-2, atol: 1.0e-3)
    end

    test "matches sklearn for shapes {8, 6}, {8, 4} and type {:f, 32} and sample_weight" do
      a =
        Nx.tensor([
          [
            0.04083956778049469,
            0.8659158945083618,
            0.3471389710903168,
            0.36252954602241516,
            0.9525835514068604,
            0.5720232725143433
          ],
          [
            0.19162702560424805,
            0.07171564549207687,
            0.4239725172519684,
            0.5658120512962341,
            0.08994761109352112,
            0.6949548125267029
          ],
          [
            0.36578986048698425,
            0.5457633137702942,
            0.8625927567481995,
            0.020068077370524406,
            0.3400953412055969,
            0.3600117564201355
          ],
          [
            0.8734534978866577,
            0.5726807117462158,
            0.1899152249097824,
            0.9930570721626282,
            0.6685937643051147,
            0.9466390609741211
          ],
          [
            0.6681521534919739,
            0.4080045819282532,
            0.09353680163621902,
            0.4131518006324768,
            0.5021317601203918,
            0.07096793502569199
          ],
          [
            0.7154380679130554,
            0.8071804046630859,
            0.8119121193885803,
            0.15046168863773346,
            0.37614867091178894,
            0.6958496570587158
          ],
          [
            0.43303877115249634,
            0.04956841096282005,
            0.47400498390197754,
            0.38280269503593445,
            0.40687817335128784,
            0.3191966116428375
          ],
          [
            0.2434559464454651,
            0.5200338363647461,
            0.1549150049686432,
            0.8125693202018738,
            0.4484674036502838,
            0.35350117087364197
          ]
        ])

      b =
        Nx.tensor([
          [0.056842684745788574, 0.9986216425895691, 0.8548102974891663, 0.8861691951751709],
          [0.0743267759680748, 0.33056357502937317, 0.8035754561424255, 0.8365055918693542],
          [0.40483880043029785, 0.3611081838607788, 0.7041659355163574, 0.18672333657741547],
          [0.7697845697402954, 0.25740090012550354, 0.14554275572299957, 0.7587448954582214],
          [0.787437379360199, 0.42764925956726074, 0.48466503620147705, 0.4724958539009094],
          [0.8312895894050598, 0.6260905265808105, 0.1480410397052765, 0.9619374871253967],
          [0.8996703624725342, 0.3709431290626526, 0.6441428065299988, 0.5562005639076233],
          [0.12500284612178802, 0.46188610792160034, 0.6846752762794495, 0.5477549433708191]
        ])

      sample_weights = [
        0.4678942561149597,
        0.9624379873275757,
        0.33467942476272583,
        0.025896426290273666,
        0.04909629747271538,
        0.28991982340812683,
        0.46588173508644104,
        0.531108558177948
      ]

      expected_coeff =
        Nx.tensor([
          [1.26270717, -0.6669458, 0.46800176, -0.11085355, 0.75226201, -0.27554435],
          [-0.36448096, 0.32648275, -0.61974275, -0.71785094, 0.20321861, 0.36756655],
          [-0.83582468, -0.26252428, -0.04247694, -0.08215458, 0.13517517, -0.12904364],
          [-0.09798772, 0.12502546, -1.26914677, -1.03593693, -0.31391357, 1.27143882]
        ])

      expected_intercept =
        Nx.tensor([
          -0.10643943421786561,
          0.7950966402865529,
          1.1103499211246088,
          1.1566576434693598
        ])

      %Scholar.Linear.LinearRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        Scholar.Linear.LinearRegression.fit(a, b, sample_weights: sample_weights)

      assert_all_close(expected_coeff, actual_coeff, rtol: 1.0e-1, atol: 1.0e-0)
      assert_all_close(expected_intercept, actual_intercept, rtol: 1.0e-1, atol: 1.0e-0)
    end

    test "test fit when fit_intercept set to false" do
      a =
        Nx.tensor([
          [
            0.7731901407241821,
            0.5813425779342651,
            0.8365984559059143,
            0.2182593196630478,
            0.06448899209499359,
            0.9420031905174255
          ],
          [
            0.6547101736068726,
            0.05023770406842232,
            0.657528281211853,
            0.24924135208129883,
            0.8238568902015686,
            0.11182288080453873
          ],
          [
            0.7693489193916321,
            0.6696648001670837,
            0.6877049803733826,
            0.08740159869194031,
            0.6053816676139832,
            0.5419610142707825
          ],
          [
            0.03419172018766403,
            0.8298202753067017,
            0.6097439527511597,
            0.0184243805706501,
            0.5578944087028503,
            0.9986271858215332
          ]
        ])

      b =
        Nx.tensor([
          0.38682249188423157,
          0.8040792346000671,
          0.8069542646408081,
          0.3620224595069885
        ])

      sample_weights = [
        0.8669093251228333,
        0.10421276837587357,
        0.996828556060791,
        0.29747673869132996
      ]

      expected_coeff =
        Nx.tensor([0.465558, 0.26869825, 0.10571962, -0.07031003, 0.56091221, -0.25331104])

      expected_intercept = Nx.tensor(0.0)

      %Scholar.Linear.LinearRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        Scholar.Linear.LinearRegression.fit(a, b,
          sample_weights: sample_weights,
          fit_intercept?: false
        )

      assert_all_close(expected_coeff, actual_coeff, rtol: 1.0e-2, atol: 1.0e-2)
      assert_all_close(expected_intercept, actual_intercept)
    end

    test "test fit when fit_intercept set to false - weights as tensor" do
      a =
        Nx.tensor([
          [
            0.7731901407241821,
            0.5813425779342651,
            0.8365984559059143,
            0.2182593196630478,
            0.06448899209499359,
            0.9420031905174255
          ],
          [
            0.6547101736068726,
            0.05023770406842232,
            0.657528281211853,
            0.24924135208129883,
            0.8238568902015686,
            0.11182288080453873
          ],
          [
            0.7693489193916321,
            0.6696648001670837,
            0.6877049803733826,
            0.08740159869194031,
            0.6053816676139832,
            0.5419610142707825
          ],
          [
            0.03419172018766403,
            0.8298202753067017,
            0.6097439527511597,
            0.0184243805706501,
            0.5578944087028503,
            0.9986271858215332
          ]
        ])

      b =
        Nx.tensor([
          0.38682249188423157,
          0.8040792346000671,
          0.8069542646408081,
          0.3620224595069885
        ])

      sample_weights = Nx.tensor([
        0.8669093251228333,
        0.10421276837587357,
        0.996828556060791,
        0.29747673869132996
      ])

      expected_coeff =
        Nx.tensor([0.465558, 0.26869825, 0.10571962, -0.07031003, 0.56091221, -0.25331104])

      expected_intercept = Nx.tensor(0.0)

      %Scholar.Linear.LinearRegression{coefficients: actual_coeff, intercept: actual_intercept} =
        Scholar.Linear.LinearRegression.fit(a, b,
          sample_weights: sample_weights,
          fit_intercept?: false
        )

      assert_all_close(expected_coeff, actual_coeff, rtol: 1.0e-2, atol: 1.0e-2)
      assert_all_close(expected_intercept, actual_intercept)
    end
  end

  describe "predict" do
    test "predict when :fit_intercept? set to true" do
      a =
        Nx.tensor([
          [
            0.7731901407241821,
            0.5813425779342651,
            0.8365984559059143,
            0.2182593196630478,
            0.06448899209499359,
            0.9420031905174255
          ],
          [
            0.6547101736068726,
            0.05023770406842232,
            0.657528281211853,
            0.24924135208129883,
            0.8238568902015686,
            0.11182288080453873
          ],
          [
            0.7693489193916321,
            0.6696648001670837,
            0.6877049803733826,
            0.08740159869194031,
            0.6053816676139832,
            0.5419610142707825
          ],
          [
            0.03419172018766403,
            0.8298202753067017,
            0.6097439527511597,
            0.0184243805706501,
            0.5578944087028503,
            0.9986271858215332
          ]
        ])

      b =
        Nx.tensor([
          0.38682249188423157,
          0.8040792346000671,
          0.8069542646408081,
          0.3620224595069885
        ])

      sample_weights = [
        0.8669093251228333,
        0.10421276837587357,
        0.996828556060791,
        0.29747673869132996
      ]

      prediction_input = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
      expected_prediction = Nx.tensor([0.16187817])

      model =
        Scholar.Linear.LinearRegression.fit(a, b,
          sample_weights: sample_weights,
          fit_intercept?: true
        )

      actual_prediction = Scholar.Linear.LinearRegression.predict(model, prediction_input)
      assert_all_close(expected_prediction, actual_prediction, rtol: 1.0e-1, atol: 1.0e-1)
    end

    test "predict when :fit_intercept? set to false" do
      a =
        Nx.tensor([
          [
            0.7731901407241821,
            0.5813425779342651,
            0.8365984559059143,
            0.2182593196630478,
            0.06448899209499359,
            0.9420031905174255
          ],
          [
            0.6547101736068726,
            0.05023770406842232,
            0.657528281211853,
            0.24924135208129883,
            0.8238568902015686,
            0.11182288080453873
          ],
          [
            0.7693489193916321,
            0.6696648001670837,
            0.6877049803733826,
            0.08740159869194031,
            0.6053816676139832,
            0.5419610142707825
          ],
          [
            0.03419172018766403,
            0.8298202753067017,
            0.6097439527511597,
            0.0184243805706501,
            0.5578944087028503,
            0.9986271858215332
          ]
        ])

      b =
        Nx.tensor([
          0.38682249188423157,
          0.8040792346000671,
          0.8069542646408081,
          0.3620224595069885
        ])

      sample_weights = [
        0.8669093251228333,
        0.10421276837587357,
        0.996828556060791,
        0.29747673869132996
      ]

      prediction_input = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
      expected_prediction = Nx.tensor([2.32356805])

      model =
        Scholar.Linear.LinearRegression.fit(a, b,
          sample_weights: sample_weights,
          fit_intercept?: false
        )

      actual_prediction = Scholar.Linear.LinearRegression.predict(model, prediction_input)
      assert_all_close(expected_prediction, actual_prediction, rtol: 1.0e-3, atol: 1.0e-3)
    end
  end
end
