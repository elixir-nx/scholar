defmodule Scholar.DiscriminantAnalysis.QuadraticTest do
  use Scholar.Case, async: true
  alias Scholar.DiscriminantAnalysis.Quadratic
  doctest Quadratic

  # Reference values come from scikit-learn 1.6.1
  # (`QuadraticDiscriminantAnalysis`), using `_decision_function` for the raw
  # class scores, since `decision_function` collapses to a single column when
  # there are two classes.

  defp two_class_x do
    Nx.tensor([
      [-2.0, -1.0],
      [-1.0, -1.0],
      [-1.0, -2.0],
      [-2.0, -2.0],
      [1.0, 1.0],
      [1.0, 2.0],
      [2.0, 1.0],
      [2.0, 2.0]
    ])
  end

  defp two_class_y, do: Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1])

  defp three_class_x do
    Nx.tensor([
      [0.0, 0.0],
      [0.5, 0.25],
      [0.25, 0.5],
      [-0.5, -0.25],
      [-0.25, -0.5],
      [4.0, 4.0],
      [4.5, 3.5],
      [3.5, 4.5],
      [4.25, 4.25],
      [3.75, 3.75],
      [-4.0, 4.0],
      [-3.5, 4.5],
      [-4.5, 3.5],
      [-4.25, 4.25],
      [-3.75, 3.75]
    ])
  end

  defp three_class_y, do: Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

  defp three_class_t, do: Nx.tensor([[0.1, 0.1], [4.0, 4.0], [-4.0, 4.0], [0.0, 4.0]])

  # Both classes share a mean and differ only in spread, so a shared covariance
  # cannot tell them apart. This is the case QDA exists for.
  defp nested_x do
    Nx.tensor([
      [0.0, 0.0],
      [0.25, 0.0],
      [-0.25, 0.0],
      [0.0, 0.25],
      [0.0, -0.25],
      [0.125, 0.125],
      [-0.125, -0.125],
      [3.0, 0.0],
      [-3.0, 0.0],
      [0.0, 3.0],
      [0.0, -3.0],
      [2.0, 2.0],
      [-2.0, -2.0],
      [2.0, -2.0],
      [-2.0, 2.0]
    ])
  end

  defp nested_y, do: Nx.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

  describe "fit" do
    test "means, priors and scalings match sklearn on two classes" do
      model = Quadratic.fit(two_class_x(), two_class_y(), num_classes: 2)

      assert_all_close(model.means, Nx.tensor([[-1.5, -1.5], [1.5, 1.5]]))
      assert_all_close(model.priors, Nx.tensor([0.5, 0.5]))

      # `eigh` pairs every eigenvalue with its eigenvector, but the order along
      # the axis is not part of its contract and QDA sums over all of them, so
      # the reference values are compared as a sorted set.
      assert_all_close(
        Nx.sort(model.scalings, axis: 1),
        Nx.tensor([[0.3333333, 0.3333333], [0.3333333, 0.3333333]])
      )
    end

    test "means, priors and scalings match sklearn on three classes" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3)

      assert_all_close(model.means, Nx.tensor([[0.0, 0.0], [4.0, 4.0], [-4.0, 4.0]]))
      assert_all_close(model.priors, Nx.tensor([0.3333333, 0.3333333, 0.3333333]))

      assert_all_close(
        Nx.sort(model.scalings, axis: 1),
        Nx.tensor([[0.03125, 0.28125], [0.0625, 0.25], [0.0625, 0.25]])
      )
    end

    test "rotations have the shape of a per-class basis" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3)

      assert Nx.shape(model.rotations) == {3, 2, 2}
      assert Nx.shape(model.scalings) == {3, 2}
      assert Nx.shape(model.means) == {3, 2}
    end

    test "the rotations reconstruct each class covariance" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3)

      scaled = Nx.multiply(model.rotations, Nx.new_axis(model.scalings, 1))

      rebuilt =
        Nx.dot(scaled, [2], [0], Nx.transpose(model.rotations, axes: [0, 2, 1]), [1], [0])

      # Covariance of class 1, computed directly.
      centered = Nx.tensor([[0.0, 0.0], [0.5, -0.5], [-0.5, 0.5], [0.25, 0.25], [-0.25, -0.25]])
      covariance = Nx.dot(centered, [0], centered, [0]) |> Nx.divide(4)

      assert_all_close(rebuilt[1], covariance)
    end

    test ":reg_param shrinks the scalings towards one, matching sklearn" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3, reg_param: 0.4)

      assert_all_close(
        Nx.sort(model.scalings, axis: 1),
        Nx.tensor([[0.41875, 0.56875], [0.4375, 0.55], [0.4375, 0.55]])
      )
    end

    test "accepts integer input" do
      x_int = Nx.tensor([[-2, -1], [-1, -1], [-1, -2], [-2, -2], [1, 1], [1, 2], [2, 1], [2, 2]])

      from_int = Quadratic.fit(x_int, two_class_y(), num_classes: 2)
      from_float = Quadratic.fit(two_class_x(), two_class_y(), num_classes: 2)

      assert_all_close(from_int.means, from_float.means)
      assert_all_close(from_int.scalings, from_float.scalings)
    end

    test "propagates f64 input" do
      model =
        Quadratic.fit(Nx.as_type(two_class_x(), :f64), two_class_y(), num_classes: 2)

      assert Nx.type(model.means) == {:f, 64}
      assert Nx.type(model.scalings) == {:f, 64}
    end
  end

  describe "decision_function" do
    test "matches sklearn on three classes" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3)

      assert_all_close(
        Quadratic.decision_function(model, three_class_t()),
        Nx.tensor([
          [1.232956, -242.379171, -255.059171],
          [-55.620378, 0.980829, -319.019171],
          [-510.731489, -319.019171, 0.980829],
          [-140.953711, -79.019171, -79.019171]
        ]),
        rtol: 1.0e-3
      )
    end

    test "matches sklearn with :reg_param" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3, reg_param: 0.4)

      assert_all_close(
        Quadratic.decision_function(model, three_class_t()),
        Nx.tensor([
          [-0.398797, -35.152069, -36.975965],
          [-28.513083, -0.386355, -66.048692],
          [-38.59017, -66.048692, -0.386355],
          [-16.96642, -16.801939, -16.801939]
        ]),
        rtol: 1.0e-3
      )
    end

    test "predict is the argmax of the decision function" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3)
      scores = Quadratic.decision_function(model, three_class_t())

      assert Quadratic.predict(model, three_class_t()) == Nx.argmax(scores, axis: 1)
    end
  end

  describe "predict" do
    test "matches sklearn on two classes" do
      model = Quadratic.fit(two_class_x(), two_class_y(), num_classes: 2)
      t = Nx.tensor([[-1.5, -1.5], [1.5, 1.5], [0.0, 0.0]])

      assert Quadratic.predict(model, t) == Nx.tensor([0, 1, 0])
    end

    test "matches sklearn on three classes" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3)

      assert Quadratic.predict(model, three_class_t()) == Nx.tensor([0, 1, 2, 1])
    end

    test "separates classes that share a mean but differ in covariance" do
      # The point of QDA: LDA pools one covariance and scores both classes
      # identically here, so it cannot beat the majority class.
      model = Quadratic.fit(nested_x(), nested_y(), num_classes: 2)

      assert Quadratic.predict(model, nested_x()) == nested_y()

      linear =
        Scholar.DiscriminantAnalysis.Linear.fit(nested_x(), nested_y(), num_classes: 2)

      linear_correct =
        Scholar.DiscriminantAnalysis.Linear.predict(linear, nested_x())
        |> Nx.equal(nested_y())
        |> Nx.sum()
        |> Nx.to_number()

      assert linear_correct < Nx.axis_size(nested_x(), 0)
    end

    test "recovers the training labels on well-separated classes" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3)

      assert Quadratic.predict(model, three_class_x()) == three_class_y()
    end
  end

  describe "predict_probability" do
    test "matches sklearn on three classes" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3)

      assert_all_close(
        Quadratic.predict_probability(model, three_class_t()),
        Nx.tensor([
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.5, 0.5]
        ])
      )
    end

    test "matches sklearn where the classes overlap" do
      model = Quadratic.fit(nested_x(), nested_y(), num_classes: 2)

      assert_all_close(
        Quadratic.predict_probability(model, Nx.tensor([[0.0, 0.0], [0.1, -0.1]])),
        Nx.tensor([[0.9940322, 0.0059678], [0.9904105, 0.0095895]])
      )
    end

    test "rows sum to one" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3)
      probs = Quadratic.predict_probability(model, three_class_t())

      assert_all_close(Nx.sum(probs, axes: [1]), Nx.broadcast(1.0, {4}))
    end
  end

  describe "jit" do
    # Required by AGENTS.md: the whole model has to survive being compiled.
    test "fit and predict run under jit_apply" do
      x = three_class_x()
      y = three_class_y()
      t = three_class_t()

      labels =
        Nx.Defn.jit_apply(
          fn x, y, t ->
            model = Quadratic.fit(x, y, num_classes: 3)
            Quadratic.predict(model, t)
          end,
          [x, y, t]
        )

      assert labels == Nx.tensor([0, 1, 2, 1])
    end

    test "decision_function runs under jit_apply" do
      model = Quadratic.fit(three_class_x(), three_class_y(), num_classes: 3)

      assert Nx.Defn.jit_apply(&Quadratic.decision_function/2, [model, three_class_t()]) ==
               Quadratic.decision_function(model, three_class_t())
    end
  end

  describe "errors" do
    test "when x is not a matrix" do
      assert_raise ArgumentError,
                   "expected x to have shape {num_samples, num_features}, got tensor with shape: {3}",
                   fn ->
                     Quadratic.fit(Nx.tensor([1, 2, 3]), Nx.tensor([0, 1, 0]), num_classes: 2)
                   end
    end

    test "when y is not a vector" do
      assert_raise ArgumentError,
                   "expected y to have shape {num_samples}, got tensor with shape: {2, 1}",
                   fn ->
                     Quadratic.fit(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), Nx.tensor([[0], [1]]),
                       num_classes: 2
                     )
                   end
    end

    test "when x and y disagree on the number of samples" do
      assert_raise ArgumentError,
                   "expected x and y to have the same number of samples, got 2 and 3",
                   fn ->
                     Quadratic.fit(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), Nx.tensor([0, 1, 0]),
                       num_classes: 2
                     )
                   end
    end

    test "when :num_classes is missing" do
      assert_raise NimbleOptions.ValidationError,
                   "required :num_classes option not found, received options: []",
                   fn -> Quadratic.fit(two_class_x(), two_class_y()) end
    end

    test "when :num_classes is not a positive integer" do
      assert_raise NimbleOptions.ValidationError,
                   "invalid value for :num_classes option: expected positive integer, got: 2.0",
                   fn -> Quadratic.fit(two_class_x(), two_class_y(), num_classes: 2.0) end
    end
  end
end
