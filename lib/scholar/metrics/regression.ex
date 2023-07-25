defmodule Scholar.Metrics.Regression do
  @moduledoc """
  Regression Metric functions.

  Metrics are used to measure the performance and compare
  the performance of any kind of regressor in
  easy-to-understand terms.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  import Nx.Defn, except: [assert_shape: 2, assert_shape_pattern: 2]
  import Scholar.Shared
  import Scholar.Metrics.Distance

  r2_schema = [
    force_finite: [
      type: :boolean,
      default: true,
      doc: """
      Flag indicating if NaN and -Inf scores resulting from constant data should be replaced with real numbers
      (1.0 if prediction is perfect, 0.0 otherwise)
      """
    ]
  ]

  @r2_schema NimbleOptions.new!(r2_schema)

  # Standard Metrics

  @doc ~S"""
  Calculates the mean absolute error of predictions
  with respect to targets.

  $$MAE = \frac{\sum_{i=1}^{n} |\hat{y_i} - y_i|}{n}$$

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]])
      iex> Scholar.Metrics.Regression.mean_absolute_error(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.5
      >
  """
  defn mean_absolute_error(y_true, y_pred) do
    assert_same_shape!(y_true, y_pred)

    (y_true - y_pred)
    |> Nx.abs()
    |> Nx.mean()
  end

  @doc ~S"""
  Calculates the mean square error of predictions
  with respect to targets.

  $$MSE = \frac{\sum_{i=1}^{n} (\hat{y_i} - y_i)^2}{n}$$

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 2.0], [0.5, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]])
      iex> Scholar.Metrics.Regression.mean_square_error(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.5625
      >
  """
  defn mean_square_error(y_true, y_pred) do
    diff = y_true - y_pred
    (diff * diff) |> Nx.mean()
  end

  @doc ~S"""
  Calculates the mean square logarithmic error of predictions
  with respect to targets.

  $$MSLE = \frac{\sum_{i=1}^{n} (\log(\hat{y_i} + 1) - \log(y_i + 1))^2}{n}$$

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]])
      iex> Scholar.Metrics.Regression.mean_square_log_error(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.24022650718688965
      >
  """
  defn mean_square_log_error(y_true, y_pred) do
    mean_square_error(Nx.log(y_true + 1), Nx.log(y_pred + 1))
  end

  @doc ~S"""
  Calculates the mean absolute percentage error of predictions
  with respect to targets. If `y_true` values are equal or close
  to zero, it returns an arbitrarily large value.

  $$MAPE = \frac{\sum_{i=1}^{n} \frac{|\hat{y_i} - y_i|}{max(\epsilon, \hat{y_i})}}{n}$$

  ## Examples

      iex> y_true = Nx.tensor([3, -0.5, 2, 7])
      iex> y_pred = Nx.tensor([2.5, 0.0, 2, 8])
      iex> Scholar.Metrics.Regression.mean_absolute_percentage_error(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.3273809552192688
      >

      iex> y_true = Nx.tensor([1.0, 0.0, 2.4, 7.0])
      iex> y_pred = Nx.tensor([1.2, 0.1, 2.4, 8.0])
      iex> Scholar.Metrics.Regression.mean_absolute_percentage_error(y_true, y_pred)
      #Nx.Tensor<
        f32
        209715.28125
      >
  """
  defn mean_absolute_percentage_error(y_true, y_pred) do
    assert_same_shape!(y_true, y_pred)

    eps =
      Nx.type(y_true)
      |> Nx.Type.merge(Nx.type(y_pred))
      |> Nx.Type.to_floating()
      |> Nx.Constants.epsilon()

    (Nx.abs(y_true - y_pred) / Nx.max(eps, Nx.abs(y_true)))
    |> Nx.mean()
  end

  @doc ~S"""
  Calculates the $R^2$ score of predictions with respect to targets.

  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

  ## Examples

      iex> y_true = Nx.tensor([3, -0.5, 2, 7], type: {:f, 32})
      iex> y_pred = Nx.tensor([2.5, 0.0, 2, 8], type: {:f, 32})
      iex> Scholar.Metrics.Regression.r2_score(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.9486081600189209
      >

      iex> Scholar.Metrics.Regression.r2_score(Nx.tensor([-2.0, -2.0, -2.0], type: :f64), Nx.tensor([-2.0, -2.0, -2.0 + 1.0e-8], type: :f64), force_finite: true)
      #Nx.Tensor<
        f64
        0.0
      >

      iex> Scholar.Metrics.Regression.r2_score(Nx.tensor([-2.0, -2.0, -2.0], type: :f64), Nx.tensor([-2.0, -2.0, -2.0 + 1.0e-8], type: :f64), force_finite: false)
      #Nx.Tensor<
        f64
        -Inf
      >

      iex> Scholar.Metrics.Regression.r2_score(Nx.tensor([-2.0, -2.0, -2.0]), Nx.tensor([-2.0, -2.0, -2.0]), force_finite: false)
      #Nx.Tensor<
        f32
        NaN
      >

      iex> Scholar.Metrics.Regression.r2_score(Nx.tensor([-2.0, -2.0, -2.0]), Nx.tensor([-2.0, -2.0, -2.0]), force_finite: true)
      #Nx.Tensor<
        f32
        1.0
      >
  """
  deftransform r2_score(y_true, y_pred, opts \\ []) do
    r2_score_n(y_true, y_pred, NimbleOptions.validate!(opts, @r2_schema))
  end

  defnp r2_score_n(y_true, y_pred, opts) do
    check_shape(y_true, y_pred)
    ssr = squared_euclidean(y_true, y_pred)

    y_mean = Nx.broadcast(Nx.mean(y_true), Nx.shape(y_true))
    sst = squared_euclidean(y_true, y_mean)

    case opts[:force_finite] do
      false ->
        infinity_mask = ssr != 0 and sst == 0
        nan_mask = ssr == 0 and sst == 0
        sst = Nx.select(sst == 0, 1, sst)
        res = Nx.select(infinity_mask, Nx.tensor(:neg_infinity), 1 - ssr / sst)
        Nx.select(nan_mask, Nx.tensor(:nan), res)

      true ->
        sst_mask = sst != 0
        ssr_mask = ssr != 0

        valid_score = ssr_mask and sst_mask

        result = Nx.broadcast(Nx.tensor(1, type: Nx.type(ssr)), ssr)
        result = Nx.select(ssr_mask and not sst_mask, 0, result)
        sst = Nx.select(not sst_mask, 1, sst)
        Nx.select(valid_score, 1 - ssr / sst, result)
    end
  end

  @doc ~S"""
  Calculates the maximum residual error.

  The residual error is defined as $$|y - \hat{y}|$$ where $y$ is a true value
  and $\hat{y}$ is a predicted value.
  This function returns the maximum residual error over all samples in the
  input: $max(|y_i - \hat{y_i}|)$. For perfect predictions, the maximum
  residual error is `0.0`.

  ## Examples

      iex> y_true = Nx.tensor([3, -0.5, 2, 7])
      iex> y_pred = Nx.tensor([2.5, 0.0, 2, 8.5])
      iex> Scholar.Metrics.Regression.max_residual_error(y_true, y_pred)
      #Nx.Tensor<
        f32
        1.5
      >
  """
  defn max_residual_error(y_true, y_pred) do
    check_shape(y_true, y_pred)
    Nx.reduce_max(Nx.abs(y_true - y_pred))
  end

  defnp check_shape(y_true, y_pred) do
    assert_rank!(y_true, 1)
    assert_same_shape!(y_true, y_pred)
  end
end
