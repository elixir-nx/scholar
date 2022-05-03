defmodule Scholar.MetricsTest do
  use ExUnit.Case, async: true

  alias Scholar.Metrics
  doctest Scholar.Metrics

  @y_true Nx.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], type: {:f, 32})
  @y_pred Nx.tensor([1.0, 2.0, 1.0, 3.0, 3.0, 3.0], type: {:f, 32})
  @cm Nx.tensor(
        [
          [1, 1, 0],
          [1, 0, 1],
          [0, 0, 2]
        ],
        type: {:s, 64}
      )
  @label Nx.tensor([1.0, 2.0, 3.0])

  test "confusion matrix" do
    assert Metrics.confusion_matrix(@y_true, @y_pred) == {@cm, @label}
  end
end
