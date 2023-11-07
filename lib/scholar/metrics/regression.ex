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
    mean_tweedie_deviance_n(y_true, y_pred, 0)
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

  @doc """
  Calculates the mean Tweedie deviance of predictions
  with respect to targets. Includes the Gaussian, Poisson,
  Gamma and inverse-Gaussian families as special cases.

  #{~S'''
  $$d(y,\mu) =
  \begin{cases}
  (y-\mu)^2, & \text{for }p=0\\\\
  2(y \log(y/\mu) + \mu - y), & \text{for }p=1\\\\
  2(\log(\mu/y) + y/\mu - 1), & \text{for }p=2\\\\
  2\left(\frac{\max(y,0)^{2-p}}{(1-p)(2-p)}-\frac{y\mu^{1-p}}{1-p}+\frac{\mu^{2-p}}{2-p}\right), & \text{for }p<0 \vee p>2
  \end{cases}$$
  '''}

  ## Examples

      iex> y_true = Nx.tensor([1, 1, 1, 1, 1, 2, 2, 1, 3, 1], type: :u32)
      iex> y_pred = Nx.tensor([2, 2, 1, 1, 2, 2, 2, 1, 3, 1], type: :u32)
      iex> Scholar.Metrics.Regression.mean_tweedie_deviance(y_true, y_pred, 1)
      #Nx.Tensor<
        f32
        0.18411168456077576
      >
  """
  defn mean_tweedie_deviance(y_true, y_pred, power) do
    mean_tweedie_deviance_n(y_true, y_pred, power)
  end

  @doc """
  Similar to `mean_tweedie_deviance/3` but raises `RuntimeError` if the
  inputs cannot be used with the given power argument.

  Note: This function cannot be used in `defn`.

  ## Examples

      iex> y_true = Nx.tensor([1, 1, 1, 1, 1, 2, 2, 1, 3, 1], type: :u32)
      iex> y_pred = Nx.tensor([2, 2, 1, 1, 2, 2, 2, 1, 3, 1], type: :u32)
      iex> Scholar.Metrics.Regression.mean_tweedie_deviance!(y_true, y_pred, 1)
      #Nx.Tensor<
        f32
        0.18411168456077576
      >
  """
  def mean_tweedie_deviance!(y_true, y_pred, power) do
    message = "mean Tweedie deviance with power=#{power} can only be used on "

    case check_tweedie_deviance_power(y_true, y_pred, power) |> Nx.to_number() do
      1 -> :ok
      2 -> raise message <> "strictly positive y_pred"
      4 -> raise message <> "non-negative y_true and strictly positive y_pred"
      5 -> raise message <> "strictly positive y_true and strictly positive y_pred"
      100 -> raise "something went wrong, branch should never appear"
    end

    mean_tweedie_deviance_n(y_true, y_pred, power)
  end

  defnp mean_tweedie_deviance_n(y_true, y_pred, power) do
    deviance =
      cond do
        power < 0 ->
          2 *
            (
              Nx.pow(max(y_true, 0), 2 - power) / ((1 - power) * (2 - power))
              -y_true * Nx.pow(y_pred, 1 - power) / (1 - power)
              +Nx.pow(y_pred, 2 - power) / (2 - power)
            )

        # Normal distribution
        power == 0 ->
          Nx.pow(y_true - y_pred, 2)

        # Poisson distribution
        power == 1 ->
          2 * (y_true * Nx.log(y_true / y_pred) + y_pred - y_true)

        # Gamma distribution
        power == 2 ->
          2 * (Nx.log(y_pred / y_true) + y_true / y_pred - 1)

        # 1 < power < 2 -> Compound Poisson distribution, non-negative with mass at zero
        # power == 3 -> Inverse-Gaussian distribution
        # power > 2 -> Stable distribution, with support on the positive reals
        true ->
          2 *
            (
              Nx.pow(y_true, 2 - power) / ((1 - power) * (2 - power))
              -y_true * Nx.pow(y_pred, 1 - power) / (1 - power)
              +Nx.pow(y_pred, 2 - power) / (2 - power)
            )
      end

    Nx.mean(deviance)
  end

  defnp check_tweedie_deviance_power(y_true, y_pred, power) do
    cond do
      power < 0 ->
        if Nx.all(y_pred > 0) do
          Nx.u8(1)
        else
          Nx.u8(2)
        end

      power == 0 ->
        Nx.u8(1)

      power >= 1 and power < 2 ->
        if Nx.all(y_true >= 0) and Nx.all(y_pred > 0) do
          Nx.u8(1)
        else
          Nx.u8(4)
        end

      power >= 2 ->
        if Nx.all(y_true > 0) and Nx.all(y_pred > 0) do
          Nx.u8(1)
        else
          Nx.u8(5)
        end

      true ->
        Nx.u8(100)
    end
  end

  @doc """
  Calculates the mean Poisson deviance of predictions
  with respect to targets.

  ## Examples

      iex> y_true = Nx.tensor([1, 1, 1, 1, 1, 2, 2, 1, 3, 1], type: :u32)
      iex> y_pred = Nx.tensor([2, 2, 1, 1, 2, 2, 2, 1, 3, 1], type: :u32)
      iex> Scholar.Metrics.Regression.mean_poisson_deviance(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.18411168456077576
      >
  """
  defn mean_poisson_deviance(y_true, y_pred) do
    mean_tweedie_deviance_n(y_true, y_pred, 1)
  end

  @doc """
  Calculates the mean Gamma deviance of predictions
  with respect to targets.

  ## Examples

      iex> y_true = Nx.tensor([1, 1, 1, 1, 1, 2, 2, 1, 3, 1], type: :u32)
      iex> y_pred = Nx.tensor([2, 2, 1, 1, 2, 2, 2, 1, 3, 1], type: :u32)
      iex> Scholar.Metrics.Regression.mean_gamma_deviance(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.115888312458992
      >
  """
  defn mean_gamma_deviance(y_true, y_pred) do
    mean_tweedie_deviance_n(y_true, y_pred, 2)
  end

  @doc """
  Calculates the $R^2$ score of predictions with respect to targets.

  #{~S'''
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  '''}

  ## Options

  #{NimbleOptions.docs(@r2_schema)}

  ## Examples

      iex> y_true = Nx.tensor([3, -0.5, 2, 7], type: {:f, 32})
      iex> y_pred = Nx.tensor([2.5, 0.0, 2, 8], type: {:f, 32})
      iex> Scholar.Metrics.Regression.r2_score(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.9486081600189209
      >

      iex> y_true = Nx.tensor([-2.0, -2.0, -2.0], type: :f64)
      iex> y_pred = Nx.tensor([-2.0, -2.0, -2.0 + 1.0e-8], type: :f64)
      iex> Scholar.Metrics.Regression.r2_score(y_true, y_pred, force_finite: true)
      #Nx.Tensor<
        f64
        0.0
      >

      iex> y_true = Nx.tensor([-2.0, -2.0, -2.0], type: :f64)
      iex> y_pred = Nx.tensor([-2.0, -2.0, -2.0 + 1.0e-8], type: :f64)
      iex> Scholar.Metrics.Regression.r2_score(y_true, y_pred, force_finite: false)
      #Nx.Tensor<
        f64
        -Inf
      >

      iex> y_true = Nx.tensor([-2.0, -2.0, -2.0])
      iex> y_pred = Nx.tensor([-2.0, -2.0, -2.0])
      iex> Scholar.Metrics.Regression.r2_score(y_true, y_pred, force_finite: false)
      #Nx.Tensor<
        f32
        NaN
      >

      iex> y_true = Nx.tensor([-2.0, -2.0, -2.0])
      iex> y_pred = Nx.tensor([-2.0, -2.0, -2.0])
      iex> Scholar.Metrics.Regression.r2_score(y_true, y_pred, force_finite: true)
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

    handle_non_finite(ssr, sst, opts)
  end

  @doc """
  Explained variance regression score function.

  Best possible score is 1.0, lower values are worse.

  ## Options

  #{NimbleOptions.docs(@r2_schema)}

  ## Examples

      iex> y_true = Nx.tensor([3, -0.5, 2, 7], type: {:f, 32})
      iex> y_pred = Nx.tensor([2.5, 0.0, 2, 8], type: {:f, 32})
      iex> Scholar.Metrics.Regression.explained_variance_score(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.9571734666824341
      >

      iex> y_true = Nx.tensor([-2.0, -2.0, -2.0], type: :f64)
      iex> y_pred = Nx.tensor([-2.0, -2.0, -2.0 + 1.0e-8], type: :f64)
      iex> Scholar.Metrics.Regression.explained_variance_score(y_true, y_pred, force_finite: true)
      #Nx.Tensor<
        f64
        0.0
      >

      iex> y_true = Nx.tensor([-2.0, -2.0, -2.0], type: :f64)
      iex> y_pred = Nx.tensor([-2.0, -2.0, -2.0 + 1.0e-8], type: :f64)
      iex> Scholar.Metrics.Regression.explained_variance_score(y_true, y_pred, force_finite: false)
      #Nx.Tensor<
        f64
        -Inf
      >

      iex> y_true = Nx.tensor([-2.0, -2.0, -2.0])
      iex> y_pred = Nx.tensor([-2.0, -2.0, -2.0])
      iex> Scholar.Metrics.Regression.explained_variance_score(y_true, y_pred, force_finite: false)
      #Nx.Tensor<
        f32
        NaN
      >

      iex> y_true = Nx.tensor([-2.0, -2.0, -2.0])
      iex> y_pred = Nx.tensor([-2.0, -2.0, -2.0])
      iex> Scholar.Metrics.Regression.explained_variance_score(y_true, y_pred, force_finite: true)
      #Nx.Tensor<
        f32
        1.0
      >
  """
  deftransform explained_variance_score(y_true, y_pred, opts \\ []) do
    explained_variance_score_n(
      y_true,
      y_pred,
      NimbleOptions.validate!(opts, @r2_schema)
    )
  end

  defnp explained_variance_score_n(y_true, y_pred, opts) do
    y_diff_avg = Nx.mean(y_true - y_pred, axes: [0])
    sample_size = Nx.axis_size(y_true, 0)

    numerator = squared_euclidean(y_true, y_pred + y_diff_avg, axes: [0]) / sample_size

    y_true_avg = Nx.mean(y_true, axes: [0])
    denominator = Nx.mean((y_true - y_true_avg) ** 2, axes: [0])
    handle_non_finite(numerator, denominator, opts)
  end

  defnp handle_non_finite(numerator, denominator, opts) do
    case opts[:force_finite] do
      false ->
        infinity_mask = numerator != 0 and denominator == 0
        nan_mask = numerator == 0 and denominator == 0
        denominator = Nx.select(denominator == 0, 1, denominator)
        res = Nx.select(infinity_mask, Nx.tensor(:neg_infinity), 1 - numerator / denominator)
        Nx.select(nan_mask, Nx.tensor(:nan), res)

      true ->
        denominator_mask = denominator != 0
        numerator_mask = numerator != 0

        valid_score = numerator_mask and denominator_mask

        result = Nx.broadcast(Nx.tensor(1, type: Nx.type(numerator)), numerator)
        result = Nx.select(numerator_mask and not denominator_mask, 0, result)
        denominator = Nx.select(denominator_mask, denominator, 1)
        Nx.select(valid_score, 1 - numerator / denominator, result)
    end
  end

  @doc """
  $D^2$ regression score function, fraction of Tweedie
  deviance explained.

  Best possible score is 1.0, lower values are worse and it
  can also be negative.

  Since it uses the mean Tweedie deviance, it also includes
  the Gaussian, Poisson, Gamma and inverse-Gaussian
  distribution families as special cases.

  ## Examples

      iex> y_true = Nx.tensor([1, 1, 1, 1, 1, 2, 2, 1, 3, 1], type: :u32)
      iex> y_pred = Nx.tensor([2, 2, 1, 1, 2, 2, 2, 1, 3, 1], type: :u32)
      iex> Scholar.Metrics.Regression.d2_tweedie_score(y_true, y_pred, 1)
      #Nx.Tensor<
        f32
        0.32202935218811035
      >
  """
  defn d2_tweedie_score(y_true, y_pred, power) do
    if Nx.size(y_pred) < 2 do
      Nx.Constants.nan()
    else
      d2_tweedie_score_n(y_true, y_pred, power)
    end
  end

  defnp d2_tweedie_score_n(y_true, y_pred, power) do
    y_true = Nx.squeeze(y_true)
    y_pred = Nx.squeeze(y_pred)

    numerator = mean_tweedie_deviance_n(y_true, y_pred, power)
    y_avg = Nx.mean(y_true)
    denominator = mean_tweedie_deviance_n(y_true, y_avg, power)

    1 - numerator / denominator
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
