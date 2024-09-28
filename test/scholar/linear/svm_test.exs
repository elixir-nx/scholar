defmodule Scholar.Linear.SVMTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.SVM
  doctest SVM

  test "Iris Data Set - multinomial classification svm test" do
    {x_train, x_test, y_train, y_test} = iris_data()

    loss_fn = fn y_pred, y_true ->
      Scholar.Linear.SVM.hinge_loss(y_pred, y_true, c: 1.0, margin: 150)
    end

    model = SVM.fit(x_train, y_train, num_classes: 3, loss_fn: loss_fn)
    res = SVM.predict(model, x_test)

    accuracy = Scholar.Metrics.Classification.accuracy(res, y_test)

    assert Nx.greater_equal(accuracy, 0.96) == Nx.u8(1)
  end

  test "test column target" do
    {x_train, x_test, y_train, y_test} = iris_data()

    loss_fn = fn y_pred, y_true ->
      Scholar.Linear.SVM.hinge_loss(y_pred, y_true, c: 1.0, margin: 150)
    end

    col_model = SVM.fit(x_train, y_train |> Nx.new_axis(-1), num_classes: 3, loss_fn: loss_fn)
    res = SVM.predict(col_model, x_test)

    model = SVM.fit(x_train, y_train, num_classes: 3, loss_fn: loss_fn)

    accuracy = Scholar.Metrics.Classification.accuracy(res, y_test)

    assert Nx.greater_equal(accuracy, 0.96) == Nx.u8(1)
    assert model == col_model
  end

  test "test fit 2 columned y data" do
    {x_train, _, y_train, _} = iris_data()

    loss_fn = fn y_pred, y_true ->
      Scholar.Linear.SVM.hinge_loss(y_pred, y_true, c: 1.0, margin: 150)
    end

    y_train = Nx.new_axis(y_train, -1)

    y_train =
      Nx.concatenate([y_train, y_train], axis: 1)

    message =
      "Scholar.Linear.SVM expected y to have shape {n_samples}, got tensor with shape: #{inspect(Nx.shape(y_train))}"

    assert_raise ArgumentError,
                 message,
                 fn -> SVM.fit(x_train, y_train, num_classes: 3, loss_fn: loss_fn) end
  end
end
