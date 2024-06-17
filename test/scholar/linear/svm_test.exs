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
  @tag :wip
  test "test column target" do
    x = Nx.tensor([[1], [2], [6], [8], [10]])
    y = Nx.tensor([1, 2, 6, 8, 10]) |> Nx.new_axis(-1)
    svm = SVM.fit(x, y)
    test = Nx.tensor([[1], [3], [4]])
    expected = Nx.tensor([1, 3, 4]) |> Nx.new_axis(-1)
    predicted = SVM.predict(svm, test)
    assert_all_close(expected, predicted, atol: 1.0e-1)
  end  
end
