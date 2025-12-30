defmodule Scholar.NaiveBayes.CategoricalTest do
  use Scholar.Case, async: true
  alias Scholar.NaiveBayes.Categorical
  doctest Categorical

  describe "fit" do
    test "fit test - all default options" do
      x = Nx.tensor([[1, 2, 3], [1, 3, 4], [2, 2, 3], [1, 1, 3], [2, 1, 4]])
      y = Nx.tensor([0, 1, 2, 1, 0])

      model = Categorical.fit(x, y, num_classes: 3)

      assert model.feature_count ==
               Nx.tensor([
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0]
                 ]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [
            [-1.6094379124341003, -0.916290731874155, -0.916290731874155, 0.0, 0.0],
            [-1.6094379124341003, -0.5108256237659905, -1.6094379124341003, 0.0, 0.0],
            [-1.3862943611198906, -1.3862943611198906, -0.6931471805599453, 0.0, 0.0]
          ],
          [
            [
              -1.791759469228055,
              -1.0986122886681096,
              -1.0986122886681096,
              -1.791759469228055,
              0.0
            ],
            [
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055,
              -1.0986122886681096,
              0.0
            ],
            [
              -1.6094379124341003,
              -1.6094379124341003,
              -0.916290731874155,
              -1.6094379124341003,
              0.0
            ]
          ],
          [
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.791759469228055,
              -1.791759469228055,
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055
            ]
          ]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-0.916290731874155, -0.916290731874155, -1.6094379124341003])

      assert_all_close(model.class_log_priors, expected_class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 2.0, 1.0])
    end

    test "fit test - :alpha set to a different value" do
      x = Nx.tensor([[1, 2, 3], [1, 3, 4], [2, 2, 3], [1, 1, 3], [2, 1, 4]])
      y = Nx.tensor([0, 1, 2, 1, 0])

      model = Categorical.fit(x, y, num_classes: 3, alpha: 1.0e-6)

      assert model.feature_count ==
               Nx.tensor([
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0]
                 ]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [
            [-14.508659238523094, -0.6931476805593202, -0.6931476805593202, 0.0, 0.0],
            [-14.508659238523094, -9.999990001618997e-07, -14.508659238523094, 0.0, 0.0],
            [-13.815513557959774, -13.815513557959774, -1.999996000066178e-06, 0.0, 0.0]
          ],
          [
            [
              -14.50865973852222,
              -0.6931481805584453,
              -0.6931481805584453,
              -14.50865973852222,
              0.0
            ],
            [
              -14.50865973852222,
              -0.6931481805584453,
              -14.50865973852222,
              -0.6931481805584453,
              0.0
            ],
            [
              -13.815514557956273,
              -13.815514557956273,
              -2.9999924999962455e-06,
              -13.815514557956273,
              0.0
            ]
          ],
          [
            [
              -14.508660238521093,
              -14.508660238521093,
              -14.508660238521093,
              -0.6931486805573204,
              -0.6931486805573204
            ],
            [
              -14.508660238521093,
              -14.508660238521093,
              -14.508660238521093,
              -0.6931486805573204,
              -0.6931486805573204
            ],
            [
              -13.815515557951773,
              -13.815515557951773,
              -13.815515557951773,
              -3.999987999934312e-06,
              -13.815515557951773
            ]
          ]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-0.916290731874155, -0.916290731874155, -1.6094379124341003])

      assert_all_close(model.class_log_priors, expected_class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 2.0, 1.0])
    end

    test "fit test - :fit_priors set to false" do
      x = Nx.tensor([[1, 2, 3], [1, 3, 4], [2, 2, 3], [1, 1, 3], [2, 1, 4]])
      y = Nx.tensor([0, 1, 2, 1, 0])

      model = Categorical.fit(x, y, num_classes: 3, fit_priors: false)

      assert model.feature_count ==
               Nx.tensor([
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0]
                 ]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [
            [-1.6094379124341003, -0.916290731874155, -0.916290731874155, 0.0, 0.0],
            [-1.6094379124341003, -0.5108256237659905, -1.6094379124341003, 0.0, 0.0],
            [-1.3862943611198906, -1.3862943611198906, -0.6931471805599453, 0.0, 0.0]
          ],
          [
            [
              -1.791759469228055,
              -1.0986122886681096,
              -1.0986122886681096,
              -1.791759469228055,
              0.0
            ],
            [
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055,
              -1.0986122886681096,
              0.0
            ],
            [
              -1.6094379124341003,
              -1.6094379124341003,
              -0.916290731874155,
              -1.6094379124341003,
              0.0
            ]
          ],
          [
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.791759469228055,
              -1.791759469228055,
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055
            ]
          ]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-1.0986122886681098, -1.0986122886681098, -1.0986122886681098])

      assert_all_close(model.class_log_priors, expected_class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 2.0, 1.0])
    end

    test "fit test - :priors are set as a list" do
      x = Nx.tensor([[1, 2, 3], [1, 3, 4], [2, 2, 3], [1, 1, 3], [2, 1, 4]])
      y = Nx.tensor([0, 1, 2, 1, 0])

      model = Categorical.fit(x, y, num_classes: 3, class_priors: [0.15, 0.25, 0.4])

      assert model.feature_count ==
               Nx.tensor([
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0]
                 ]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [
            [-1.6094379124341003, -0.916290731874155, -0.916290731874155, 0.0, 0.0],
            [-1.6094379124341003, -0.5108256237659905, -1.6094379124341003, 0.0, 0.0],
            [-1.3862943611198906, -1.3862943611198906, -0.6931471805599453, 0.0, 0.0]
          ],
          [
            [
              -1.791759469228055,
              -1.0986122886681096,
              -1.0986122886681096,
              -1.791759469228055,
              0.0
            ],
            [
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055,
              -1.0986122886681096,
              0.0
            ],
            [
              -1.6094379124341003,
              -1.6094379124341003,
              -0.916290731874155,
              -1.6094379124341003,
              0.0
            ]
          ],
          [
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.791759469228055,
              -1.791759469228055,
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055
            ]
          ]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-1.8971199848858813, -1.3862943611198906, -0.916290731874155])

      assert_all_close(model.class_log_priors, expected_class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 2.0, 1.0])
    end

    test "fit test - :priors are set as a tensor" do
      x = Nx.tensor([[1, 2, 3], [1, 3, 4], [2, 2, 3], [1, 1, 3], [2, 1, 4]])
      y = Nx.tensor([0, 1, 2, 1, 0])

      model = Categorical.fit(x, y, num_classes: 3, class_priors: Nx.tensor([0.15, 0.25, 0.4]))

      assert model.feature_count ==
               Nx.tensor([
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0]
                 ]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [
            [-1.6094379124341003, -0.916290731874155, -0.916290731874155, 0.0, 0.0],
            [-1.6094379124341003, -0.5108256237659905, -1.6094379124341003, 0.0, 0.0],
            [-1.3862943611198906, -1.3862943611198906, -0.6931471805599453, 0.0, 0.0]
          ],
          [
            [
              -1.791759469228055,
              -1.0986122886681096,
              -1.0986122886681096,
              -1.791759469228055,
              0.0
            ],
            [
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055,
              -1.0986122886681096,
              0.0
            ],
            [
              -1.6094379124341003,
              -1.6094379124341003,
              -0.916290731874155,
              -1.6094379124341003,
              0.0
            ]
          ],
          [
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.791759469228055,
              -1.791759469228055,
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055
            ]
          ]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-1.8971199848858813, -1.3862943611198906, -0.916290731874155])

      assert_all_close(model.class_log_priors, expected_class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 2.0, 1.0])
    end

    test "fit test - :sample_weights are set as a list" do
      x = Nx.tensor([[1, 2, 3], [1, 3, 4], [2, 2, 3], [1, 1, 3], [2, 1, 4]])
      y = Nx.tensor([0, 1, 2, 1, 0])

      model = Categorical.fit(x, y, num_classes: 3, sample_weights: [1.5, 4, 2, 7, 4])

      assert model.feature_count ==
               Nx.tensor([
                 [
                   [0.0, 1.5, 4.0, 0.0, 0.0],
                   [0.0, 11.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 2.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 4.0, 1.5, 0.0, 0.0],
                   [0.0, 7.0, 0.0, 4.0, 0.0],
                   [0.0, 0.0, 2.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 0.0, 0.0, 1.5, 4.0],
                   [0.0, 0.0, 0.0, 7.0, 4.0],
                   [0.0, 0.0, 0.0, 2.0, 0.0]
                 ]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [
            [-2.140066146850586, -1.2237753868103027, -0.5306282043457031, 0.0, 0.0],
            [-2.6390573978424072, -0.15415072441101074, -2.6390573978424072, 0.0, 0.0],
            [-1.6094379425048828, -1.6094379425048828, -0.5108256340026855, 0.0, 0.0]
          ],
          [
            [
              -2.2512917518615723,
              -0.6418538093566895,
              -1.335000991821289,
              -2.2512917518615723,
              0.0
            ],
            [-2.70805025100708, -0.6286087036132812, -2.70805025100708, -1.0986123085021973, 0.0],
            [
              -1.7917594909667969,
              -1.7917594909667969,
              -0.6931471824645996,
              -1.7917594909667969,
              0.0
            ]
          ],
          [
            [
              -2.3513753414154053,
              -2.3513753414154053,
              -2.3513753414154053,
              -1.435084581375122,
              -0.7419373989105225
            ],
            [
              -2.7725887298583984,
              -2.7725887298583984,
              -2.7725887298583984,
              -0.6931471824645996,
              -1.1631507873535156
            ],
            [
              -1.945910096168518,
              -1.945910096168518,
              -1.945910096168518,
              -0.8472977876663208,
              -1.945910096168518
            ]
          ]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-1.2130225896835327, -0.5198752880096436, -2.224623441696167])

      assert_all_close(model.class_log_priors, expected_class_log_priors)

      assert model.class_count == Nx.tensor([5.5, 11.0, 2.0])
    end

    test "fit test - :sample_weights are set as a tensor" do
      x = Nx.tensor([[1, 2, 3], [1, 3, 4], [2, 2, 3], [1, 1, 3], [2, 1, 4]])
      y = Nx.tensor([0, 1, 2, 1, 0])

      model = Categorical.fit(x, y, num_classes: 3, sample_weights: Nx.tensor([1.5, 4, 2, 7, 4]))

      assert model.feature_count ==
               Nx.tensor([
                 [
                   [0.0, 1.5, 4.0, 0.0, 0.0],
                   [0.0, 11.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 2.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 4.0, 1.5, 0.0, 0.0],
                   [0.0, 7.0, 0.0, 4.0, 0.0],
                   [0.0, 0.0, 2.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 0.0, 0.0, 1.5, 4.0],
                   [0.0, 0.0, 0.0, 7.0, 4.0],
                   [0.0, 0.0, 0.0, 2.0, 0.0]
                 ]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [
            [-2.140066146850586, -1.2237753868103027, -0.5306282043457031, 0.0, 0.0],
            [-2.6390573978424072, -0.15415072441101074, -2.6390573978424072, 0.0, 0.0],
            [-1.6094379425048828, -1.6094379425048828, -0.5108256340026855, 0.0, 0.0]
          ],
          [
            [
              -2.2512917518615723,
              -0.6418538093566895,
              -1.335000991821289,
              -2.2512917518615723,
              0.0
            ],
            [-2.70805025100708, -0.6286087036132812, -2.70805025100708, -1.0986123085021973, 0.0],
            [
              -1.7917594909667969,
              -1.7917594909667969,
              -0.6931471824645996,
              -1.7917594909667969,
              0.0
            ]
          ],
          [
            [
              -2.3513753414154053,
              -2.3513753414154053,
              -2.3513753414154053,
              -1.435084581375122,
              -0.7419373989105225
            ],
            [
              -2.7725887298583984,
              -2.7725887298583984,
              -2.7725887298583984,
              -0.6931471824645996,
              -1.1631507873535156
            ],
            [
              -1.945910096168518,
              -1.945910096168518,
              -1.945910096168518,
              -0.8472977876663208,
              -1.945910096168518
            ]
          ]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-1.2130225896835327, -0.5198752880096436, -2.224623441696167])

      assert_all_close(model.class_log_priors, expected_class_log_priors)

      assert model.class_count == Nx.tensor([5.5, 11.0, 2.0])
    end

    test "fit test - :min_categories are set as a list" do
      x = Nx.tensor([[1, 2, 3], [1, 3, 4], [2, 2, 3], [1, 1, 3], [2, 1, 4]])
      y = Nx.tensor([0, 1, 2, 1, 0])

      model = Categorical.fit(x, y, num_classes: 3, min_categories: [5, 5, 5])

      assert model.feature_count ==
               Nx.tensor([
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0]]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [
            [
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368,
              -1.9459101490553132,
              -1.9459101490553132
            ],
            [
              -1.9459101490553132,
              -0.8472978603872034,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132
            ],
            [
              -1.791759469228055,
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055,
              -1.791759469228055
            ]
          ],
          [
            [
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368,
              -1.9459101490553132,
              -1.9459101490553132
            ],
            [
              -1.9459101490553132,
              -1.252762968495368,
              -1.9459101490553132,
              -1.252762968495368,
              -1.9459101490553132
            ],
            [
              -1.791759469228055,
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055,
              -1.791759469228055
            ]
          ],
          [
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.791759469228055,
              -1.791759469228055,
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055
            ]
          ]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-0.916290731874155, -0.916290731874155, -1.6094379124341003])

      assert_all_close(model.class_log_priors, expected_class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 2.0, 1.0])
    end

    test "fit test - :min_categories are set as a tensor" do
      x = Nx.tensor([[1, 2, 3], [1, 3, 4], [2, 2, 3], [1, 1, 3], [2, 1, 4]])
      y = Nx.tensor([0, 1, 2, 1, 0])

      model = Categorical.fit(x, y, num_classes: 3, min_categories: Nx.tensor([5.0, 5.0, 5.0]))

      assert model.feature_count ==
               Nx.tensor([
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [
                   [0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]
                 ],
                 [[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0]]
               ])

      expected_feature_log_probability =
        Nx.tensor([
          [
            [
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368,
              -1.9459101490553132,
              -1.9459101490553132
            ],
            [
              -1.9459101490553132,
              -0.8472978603872034,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132
            ],
            [
              -1.791759469228055,
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055,
              -1.791759469228055
            ]
          ],
          [
            [
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368,
              -1.9459101490553132,
              -1.9459101490553132
            ],
            [
              -1.9459101490553132,
              -1.252762968495368,
              -1.9459101490553132,
              -1.252762968495368,
              -1.9459101490553132
            ],
            [
              -1.791759469228055,
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055,
              -1.791759469228055
            ]
          ],
          [
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.9459101490553132,
              -1.9459101490553132,
              -1.9459101490553132,
              -1.252762968495368,
              -1.252762968495368
            ],
            [
              -1.791759469228055,
              -1.791759469228055,
              -1.791759469228055,
              -1.0986122886681096,
              -1.791759469228055
            ]
          ]
        ])

      assert_all_close(model.feature_log_probability, expected_feature_log_probability)

      expected_class_log_priors =
        Nx.tensor([-0.916290731874155, -0.916290731874155, -1.6094379124341003])

      assert_all_close(model.class_log_priors, expected_class_log_priors)

      assert model.class_count == Nx.tensor([2.0, 2.0, 1.0])
    end
  end

  describe "errors" do
    test "wrong input rank" do
      assert_raise ArgumentError,
                   "expected x to have shape {num_samples, num_features}, got tensor with shape: {4}",
                   fn ->
                     Categorical.fit(
                       Nx.tensor([1, 2, 5, 8]),
                       Nx.tensor([1, 2, 3, 4]),
                       num_classes: 4
                     )
                   end
    end

    test "wrong target rank" do
      assert_raise ArgumentError,
                   "expected y to have shape {num_samples}, got tensor with shape: {1, 4}",
                   fn ->
                     Categorical.fit(
                       Nx.tensor([[1, 2, 5, 8]]),
                       Nx.tensor([[1, 2, 3, 4]]),
                       num_classes: 4
                     )
                   end
    end

    test "wrong input shape" do
      assert_raise ArgumentError,
                   "expected first dimension of x and y to be of same size, got: 1 and 4",
                   fn ->
                     Categorical.fit(
                       Nx.tensor([[1, 2, 5, 8]]),
                       Nx.tensor([1, 2, 3, 4]),
                       num_classes: 4
                     )
                   end
    end

    test "wrong prior size" do
      assert_raise ArgumentError,
                   "expected class_priors to be list of length num_classes = 2, got: 3",
                   fn ->
                     Categorical.fit(
                       Nx.tensor([[1, 2, 5, 8], [2, 5, 7, 3]]),
                       Nx.tensor([1, 0]),
                       num_classes: 2,
                       class_priors: [0.4, 0.4, 0.2]
                     )
                   end
    end

    test "wrong sample_weights size" do
      assert_raise ArgumentError,
                   "expected sample_weights to be list of length num_samples = 2, got: 3",
                   fn ->
                     Categorical.fit(
                       Nx.tensor([[1, 2, 5, 8], [2, 5, 7, 3]]),
                       Nx.tensor([1, 0]),
                       num_classes: 2,
                       sample_weights: [0.4, 0.4, 0.2]
                     )
                   end
    end

    test "wrong min_categories size" do
      assert_raise ArgumentError,
                   "expected min_categories to be list of length num_features = 4, got: 3",
                   fn ->
                     Categorical.fit(
                       Nx.tensor([[1, 2, 5, 8], [2, 5, 7, 3]]),
                       Nx.tensor([1, 0]),
                       num_classes: 2,
                       min_categories: [5.0, 5.0, 5.0]
                     )
                   end
    end

    test "wrong alpha size" do
      assert_raise ArgumentError,
                   "when alpha is list it should have length equal to num_features = 4, got: 3",
                   fn ->
                     Categorical.fit(
                       Nx.tensor([[1, 2, 5, 8], [2, 5, 7, 3]]),
                       Nx.tensor([1, 0]),
                       num_classes: 2,
                       alpha: [0.4, 0.4, 0.2]
                     )
                   end
    end

    test "wrong input shape in training process" do
      assert_raise ArgumentError,
                   "expected x to have same second dimension as data used for fitting model, got: 5 for x and 6 for training data",
                   fn ->
                     x = Nx.iota({5, 6})
                     y = Nx.tensor([1, 4, 3, 4, 5])

                     model = Categorical.fit(x, y, num_classes: 6)

                     x_test = Nx.tensor([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]])

                     Categorical.predict(model, x_test, Nx.tensor([0, 1, 2, 3, 4, 5]))
                   end
    end
  end

  describe "predict" do
    test "predicts classes correctly for new data" do
      x = Nx.iota({5, 6})
      y = Nx.tensor([1, 4, 3, 4, 5])

      model = Categorical.fit(x, y, num_classes: 6)

      x_test = Nx.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]])

      minus_infinity = Nx.Constants.infinity() |> Nx.negate() |> Nx.to_number()

      predictions = Categorical.predict(model, x_test, Nx.tensor([0, 1, 2, 3, 4, 5]))

      assert predictions == Nx.tensor([4, 1])

      log_probability =
        Categorical.predict_log_probability(model, x_test)

      assert_all_close(
        log_probability,
        Nx.tensor([
          [
            minus_infinity,
            -1.5314763709643877,
            minus_infinity,
            -1.5314763709643877,
            -1.0459685551826894,
            -1.5314763709643877
          ],
          [
            minus_infinity,
            -1.0340737675305398,
            minus_infinity,
            -1.7272209480904834,
            -1.241713132308785,
            -1.7272209480904834
          ]
        ])
      )

      probability =
        Categorical.predict_probability(model, x_test)

      assert_all_close(
        probability,
        Nx.tensor([
          [
            0.0,
            0.2162162162162164,
            0.0,
            0.2162162162162164,
            0.35135135135135076,
            0.2162162162162164
          ],
          [
            0.0,
            0.35555555555555507,
            0.0,
            0.17777777777777784,
            0.2888888888888883,
            0.17777777777777784
          ]
        ])
      )

      joint_log_probability =
        Categorical.predict_joint_log_probability(model, x_test)

      assert_all_close(
        joint_log_probability,
        Nx.tensor([
          [
            minus_infinity,
            -21.69805624276889,
            minus_infinity,
            -21.69805624276889,
            -21.21254842698719,
            -21.69805624276889
          ],
          [
            minus_infinity,
            -21.004909062208945,
            minus_infinity,
            -21.69805624276889,
            -21.21254842698719,
            -21.69805624276889
          ]
        ])
      )
    end
  end
end
