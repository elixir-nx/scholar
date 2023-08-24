defmodule Scholar.Linear.SVMTest do
  use Scholar.Case, async: true
  alias Scholar.Linear.SVM
  doctest SVM

  test "Iris Data Set - multinomial classification svm test" do
    Nx.Defn.default_options(compiler: EXLA)

    {x_train, x_test, y_train, y_test} = Datasets.get(:iris) |> Nx.backend_transfer(EXLA.Backend)
    y_train = Nx.dot(y_train, Nx.iota({3}, backend: EXLA.Backend))
    y_test = Nx.dot(y_test, Nx.iota({3}, backend: EXLA.Backend))

    x_train = Scholar.Preprocessing.standard_scale(x_train)
    x_test = Scholar.Preprocessing.standard_scale(x_test)

    model = SVM.fit(x_train, y_train, num_classes: 3, margin: 10)
    res = SVM.predict(model, x_test)

    accuracy = Scholar.Metrics.Classification.accuracy(res, y_test)

    assert Nx.greater_equal(Nx.backend_transfer(accuracy), 0.89) == Nx.u8(1)
  end
end
