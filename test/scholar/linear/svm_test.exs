defmodule Scholar.Linear.SVMTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.SVM
  doctest SVM

  test "Iris Data Set - multinomial classification svm test" do
    {x_train, x_test, y_train, y_test} = iris_data()

    model = SVM.fit(x_train, y_train, num_classes: 3, margin: 150)
    res = SVM.predict(model, x_test)

    accuracy = Scholar.Metrics.Classification.accuracy(res, y_test)

    assert Nx.greater_equal(accuracy, 0.96) == Nx.u8(1)
  end
end
