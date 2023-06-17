defmodule Scholar.Integrate.SimpsonTest do
  use Scholar.Case, async: true

  doctest Scholar.Integrate.Simpson

  # Since this algorithm has corrner cases for small shapes, we need to test
  # it with a variety of shapes.

  describe "simpson_uniform" do
    test "y - 1D 3 elements, default options" do
      y = Nx.tensor([0.50489939, 0.6058868, 0.85461666])
      expected_result = Nx.tensor(1.261021083333333)
      assert_all_close(Scholar.Integrate.Simpson.simpson_uniform(y), expected_result)
    end

    test "y - 1D 4 elements, default options" do
      y = Nx.tensor([0.50489939, 0.6058868, 0.85461666, 0.7607792])
      expected_result = Nx.tensor(2.109578166170962)
      assert_all_close(Scholar.Integrate.Simpson.simpson_uniform(y), expected_result)
    end

    test "y - 1D 5 elements, default options" do
      y = Nx.tensor([0.50489939, 0.6058868, 0.85461666, 0.7607792, 0.255422])
      expected_result = Nx.tensor(2.645406236666666)
      assert_all_close(Scholar.Integrate.Simpson.simpson_uniform(y), expected_result)
    end

    test "y - 2D 2x3 elements, default options" do
      y = Nx.tensor([[0.85446791, 0.71695117, 0.02815885], [0.98594984, 0.8318059, 0.55712403]])
      expected_result = Nx.tensor([1.25014381, 1.62343249])
      assert_all_close(Scholar.Integrate.Simpson.simpson_uniform(y), expected_result)
    end

    test "y - 2D 2x3 elements, axis: 0" do
      y = Nx.tensor([[0.85446791, 0.71695117, 0.02815885], [0.98594984, 0.8318059, 0.55712403]])
      expected_result = Nx.tensor([0.92020888, 0.77437853, 0.29264144])
      assert_all_close(Scholar.Integrate.Simpson.simpson_uniform(y, axis: 0), expected_result)
    end

    test "y - 2D 4x5 elements, default options" do
      y =
        Nx.tensor([
          [0.43309712, 0.77904949, 0.18364901, 0.31322696, 0.26491763],
          [0.33316243, 0.15002905, 0.21140855, 0.12291938, 0.98921421],
          [0.38665981, 0.76903673, 0.97861651, 0.37014787, 0.78876061],
          [0.04867897, 0.60992699, 0.1101239, 0.57535037, 0.89041301]
        ])

      expected_result = Nx.tensor([1.81147286, 0.94566249, 2.56313061, 1.96681641])
      assert_all_close(Scholar.Integrate.Simpson.simpson_uniform(y), expected_result)
    end

    test "y - 2D 4x5 elements, axis: 0" do
      y =
        Nx.tensor([
          [0.43309712, 0.77904949, 0.18364901, 0.31322696, 0.26491763],
          [0.33316243, 0.15002905, 0.21140855, 0.12291938, 0.98921421],
          [0.38665981, 0.76903673, 0.97861651, 0.37014787, 0.78876061],
          [0.04867897, 0.60992699, 0.1101239, 0.57535037, 0.89041301]
        ])

      expected_result = Nx.tensor([0.98054747, 1.5743948, 1.41159919, 0.90439673, 2.40752716])
      assert_all_close(Scholar.Integrate.Simpson.simpson_uniform(y, axis: 0), expected_result)
    end

    test "y - 2D 4x5 elements, axis: 0, dx: 2" do
      y =
        Nx.tensor([
          [0.43309712, 0.77904949, 0.18364901, 0.31322696, 0.26491763],
          [0.33316243, 0.15002905, 0.21140855, 0.12291938, 0.98921421],
          [0.38665981, 0.76903673, 0.97861651, 0.37014787, 0.78876061],
          [0.04867897, 0.60992699, 0.1101239, 0.57535037, 0.89041301]
        ])

      expected_result = Nx.tensor([1.96109494, 3.14878961, 2.82319839, 1.80879347, 4.81505433])

      assert_all_close(
        Scholar.Integrate.Simpson.simpson_uniform(y, axis: 0, dx: 2),
        expected_result
      )
    end

    test "y - 2D 4x5 elements, axis: 0, dx: 2, keep_axis: true" do
      y =
        Nx.tensor([
          [0.43309712, 0.77904949, 0.18364901, 0.31322696, 0.26491763],
          [0.33316243, 0.15002905, 0.21140855, 0.12291938, 0.98921421],
          [0.38665981, 0.76903673, 0.97861651, 0.37014787, 0.78876061],
          [0.04867897, 0.60992699, 0.1101239, 0.57535037, 0.89041301]
        ])

      expected_result = Nx.tensor([[1.96109494, 3.14878961, 2.82319839, 1.80879347, 4.81505433]])

      assert_all_close(
        Scholar.Integrate.Simpson.simpson_uniform(y, axis: 0, dx: 2, keep_axis: true),
        expected_result
      )
    end

    test "y - 3D 4x5x3 elements, axis: 1, dx: 2" do
      y =
        Nx.tensor([
          [
            [0.23264723, 0.42379848, 0.87790989],
            [0.98550514, 0.36063556, 0.32022114],
            [0.77996481, 0.25315318, 0.30530338],
            [0.98436553, 0.24898287, 0.37874696],
            [0.51685348, 0.06336619, 0.94235288]
          ],
          [
            [0.37633484, 0.33653536, 0.97386364],
            [0.2939261, 0.03227474, 0.35725894],
            [0.06391706, 0.1054172, 0.55916931],
            [0.19110734, 0.11700674, 0.68538223],
            [0.93515427, 0.46097941, 0.43542597]
          ],
          [
            [0.39788368, 0.29688467, 0.93306917],
            [0.01651667, 0.11868639, 0.77870915],
            [0.98668355, 0.24792443, 0.9798144],
            [0.046029, 0.79311591, 0.82452232],
            [0.5104284, 0.96706657, 0.36747779]
          ],
          [
            [0.95503247, 0.23269009, 0.57466964],
            [0.34055117, 0.3510329, 0.58890443],
            [0.27281503, 0.77761174, 0.15183282],
            [0.98733652, 0.56249535, 0.23378592],
            [0.49852633, 0.8152524, 0.20584113]
          ]
        ])

      expected_result =
        Nx.tensor([
          [6.79260866, 2.28796315, 3.48449461],
          [2.25297135, 1.07031673, 4.46546194],
          [2.08790791, 3.60467285, 6.44873441],
          [4.87382641, 4.17151932, 2.9166252]
        ])

      assert_all_close(
        Scholar.Integrate.Simpson.simpson_uniform(y, axis: 1, dx: 2),
        expected_result
      )
    end
  end

  describe "simpson" do
    test "y - 1D 3 elements, x - 1D 3 elements, default options" do
      y = Nx.tensor([0.50489939, 0.6058868, 0.85461666])
      x = Nx.tensor([0.21978049709980663, 0.29471454397376906, 0.44992644148022976])
      expected_result = Nx.tensor(0.15419391318593953)
      assert_all_close(Scholar.Integrate.Simpson.simpson(y, x), expected_result)
    end

    test "y - 1D 4 elements, x - 1D 4 elements, default options" do
      y = Nx.tensor([0.50489939, 0.6058868, 0.85461666, 0.7607792])

      x =
        Nx.tensor([
          0.16656364452415529,
          0.24921784409466585,
          0.4309042300540592,
          0.6264657771262101
        ])

      expected_result = Nx.tensor(0.3417364215450491)
      assert_all_close(Scholar.Integrate.Simpson.simpson(y, x), expected_result)
    end

    test "y - 1D 5 elements, x - 1D 5 elements, default options" do
      y = Nx.tensor([0.50489939, 0.6058868, 0.85461666, 0.7607792, 0.255422])

      x =
        Nx.tensor([
          0.12317738881481222,
          0.45630325440787045,
          0.5012925717498343,
          0.5676959700499795,
          0.6909899802922868
        ])

      expected_result = Nx.tensor(0.2539041132135682)
      assert_all_close(Scholar.Integrate.Simpson.simpson(y, x), expected_result)
    end

    test "y - 2D 2x3 elements, x - 2D 2x3 elements, default options" do
      y = Nx.tensor([[0.85446791, 0.71695117, 0.02815885], [0.98594984, 0.8318059, 0.55712403]])
      x = Nx.tensor([[0.46890615, 0.49058314, 0.81501009], [0.18575926, 0.84769413, 0.90308324]])
      expected_result = Nx.tensor([0.06847349, 0.95876125])
      assert_all_close(Scholar.Integrate.Simpson.simpson(y, x), expected_result)
    end

    test "y - 2D 2x3 elements, x - 2D 2x3 elements, axis: 0" do
      y = Nx.tensor([[0.85446791, 0.71695117, 0.02815885], [0.98594984, 0.8318059, 0.55712403]])
      x = Nx.tensor([[0.46890615, 0.49058314, 0.81501009], [0.18575926, 0.84769413, 0.90308324]])
      expected_result = Nx.tensor([-0.26055429, 0.27653908, 0.02577385])
      assert_all_close(Scholar.Integrate.Simpson.simpson(y, x, axis: 0), expected_result)
    end

    test "y - 2D 4x5 elements, x - 2D 4x5 elements, default options" do
      y =
        Nx.tensor([
          [0.43309712, 0.77904949, 0.18364901, 0.31322696, 0.26491763],
          [0.33316243, 0.15002905, 0.21140855, 0.12291938, 0.98921421],
          [0.38665981, 0.76903673, 0.97861651, 0.37014787, 0.78876061],
          [0.04867897, 0.60992699, 0.1101239, 0.57535037, 0.89041301]
        ])

      x =
        Nx.tensor([
          [0.26637336, 0.26991614, 0.31952901, 0.45237751, 0.64462463],
          [0.42041206, 0.42626697, 0.78027303, 0.78748166, 0.86583244],
          [0.11605222, 0.15723427, 0.22862411, 0.67573644, 0.70960217],
          [0.17673384, 0.21032491, 0.21856944, 0.39670414, 0.58508576]
        ])

      expected_result = Nx.tensor([0.16254933, -0.55795823, -0.01377725, 0.23026339])
      assert_all_close(Scholar.Integrate.Simpson.simpson(y, x), expected_result)
    end

    test "y - 2D 4x5 elements, x - 2D 4x5 elements, axis: 0" do
      y =
        Nx.tensor([
          [0.43309712, 0.77904949, 0.18364901, 0.31322696, 0.26491763],
          [0.33316243, 0.15002905, 0.21140855, 0.12291938, 0.98921421],
          [0.38665981, 0.76903673, 0.97861651, 0.37014787, 0.78876061],
          [0.04867897, 0.60992699, 0.1101239, 0.57535037, 0.89041301]
        ])

      x =
        Nx.tensor([
          [0.26637336, 0.26991614, 0.31952901, 0.45237751, 0.64462463],
          [0.42041206, 0.42626697, 0.78027303, 0.78748166, 0.86583244],
          [0.11605222, 0.15723427, 0.22862411, 0.67573644, 0.70960217],
          [0.17673384, 0.21032491, 0.21856944, 0.39670414, 0.58508576]
        ])

      expected_result =
        Nx.tensor([0.00784445, -0.02913629, -2.33543955, -0.07143407, -0.08323345])

      assert_all_close(Scholar.Integrate.Simpson.simpson(y, x, axis: 0), expected_result)
    end

    test "y - 2D 4x5 elements, x - 2D 4x5 elements, axis: 0, keep_axis: true" do
      y =
        Nx.tensor([
          [0.43309712, 0.77904949, 0.18364901, 0.31322696, 0.26491763],
          [0.33316243, 0.15002905, 0.21140855, 0.12291938, 0.98921421],
          [0.38665981, 0.76903673, 0.97861651, 0.37014787, 0.78876061],
          [0.04867897, 0.60992699, 0.1101239, 0.57535037, 0.89041301]
        ])

      x =
        Nx.tensor([
          [0.26637336, 0.26991614, 0.31952901, 0.45237751, 0.64462463],
          [0.42041206, 0.42626697, 0.78027303, 0.78748166, 0.86583244],
          [0.11605222, 0.15723427, 0.22862411, 0.67573644, 0.70960217],
          [0.17673384, 0.21032491, 0.21856944, 0.39670414, 0.58508576]
        ])

      expected_result =
        Nx.tensor([[0.00784445, -0.02913629, -2.33543955, -0.07143407, -0.08323345]])

      assert_all_close(
        Scholar.Integrate.Simpson.simpson(y, x, axis: 0, keep_axis: true),
        expected_result
      )
    end

    test "y - 2D 4x5 elements, x - 2D 1x5 elements, axis: 1" do
      y =
        Nx.tensor([
          [0.43309712, 0.77904949, 0.18364901, 0.31322696, 0.26491763],
          [0.33316243, 0.15002905, 0.21140855, 0.12291938, 0.98921421],
          [0.38665981, 0.76903673, 0.97861651, 0.37014787, 0.78876061],
          [0.04867897, 0.60992699, 0.1101239, 0.57535037, 0.89041301]
        ])

      x = Nx.tensor([[0.03565908, 0.05843005, 0.05944107, 0.50542287, 0.57079342]])

      expected_result = Nx.tensor([0.22383512, -0.27920952, 0.11162089, 0.14204503])
      assert_all_close(Scholar.Integrate.Simpson.simpson(y, x, axis: 1), expected_result)
    end

    test "y - 2D 4x5 elements, x - 2D 4x5 elements, axis: 0, even: :first" do
      y =
        Nx.tensor([
          [0.43309712, 0.77904949, 0.18364901, 0.31322696, 0.26491763],
          [0.33316243, 0.15002905, 0.21140855, 0.12291938, 0.98921421],
          [0.38665981, 0.76903673, 0.97861651, 0.37014787, 0.78876061],
          [0.04867897, 0.60992699, 0.1101239, 0.57535037, 0.89041301]
        ])

      x =
        Nx.tensor([
          [0.26637336, 0.26991614, 0.31952901, 0.45237751, 0.64462463],
          [0.42041206, 0.42626697, 0.78027303, 0.78748166, 0.86583244],
          [0.11605222, 0.15723427, 0.22862411, 0.67573644, 0.70960217],
          [0.17673384, 0.21032491, 0.21856944, 0.39670414, 0.58508576]
        ])

      expected_result =
        Nx.tensor([-0.05018619, -0.05426026, -0.05630284, -0.04191934, -0.06890845])

      assert_all_close(
        Scholar.Integrate.Simpson.simpson(y, x, axis: 0, even: :first),
        expected_result
      )
    end

    test "y - 2D 4x5 elements, x - 2D 4x5 elements, axis: 0, even: :last" do
      y =
        Nx.tensor([
          [0.43309712, 0.77904949, 0.18364901, 0.31322696, 0.26491763],
          [0.33316243, 0.15002905, 0.21140855, 0.12291938, 0.98921421],
          [0.38665981, 0.76903673, 0.97861651, 0.37014787, 0.78876061],
          [0.04867897, 0.60992699, 0.1101239, 0.57535037, 0.89041301]
        ])

      x =
        Nx.tensor([
          [0.26637336, 0.26991614, 0.31952901, 0.45237751, 0.64462463],
          [0.42041206, 0.42626697, 0.78027303, 0.78748166, 0.86583244],
          [0.11605222, 0.15723427, 0.22862411, 0.67573644, 0.70960217],
          [0.17673384, 0.21032491, 0.21856944, 0.39670414, 0.58508576]
        ])

      expected_result =
        Nx.tensor([
          6.58750849e-02,
          -4.01231254e-03,
          -4.61457559e+00,
          -1.00948797e-01,
          -9.75584499e-02
        ])

      assert_all_close(
        Scholar.Integrate.Simpson.simpson(y, x, axis: 0, even: :last),
        expected_result
      )
    end
  end
end
