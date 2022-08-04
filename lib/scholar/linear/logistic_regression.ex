defmodule Scholar.Linear.LogisticRegression do
  @moduledoc """
  Logistic regression in both binary and multinomial variants
  """
  import Nx.Defn
  alias __MODULE__

  @derive {Nx.Container, containers: [:coefficients, :bias], keep: [:mode]}
  defstruct [:coefficients, :bias, :mode]

  @doc """
  Fits a logistic regression model for sample inputs `x` and
  sample targets `y`. Depending on number of classes the function chooses
  either binary or multinomial logistic regression.
  """
  defn fit(x, y, opts \\ []) do
    fit_verify(x, y, opts)
    if opts[:num_classes] < 3, do: fit_binary(x, y, opts), else: fit_multinomial(x, y, opts)
  end

  # Function checks validity of the provided data

  deftransformp fit_verify(x, y, opts) do
    if !is_integer(opts[:num_classes]) or opts[:num_classes] < 1 do
      raise ArgumentError, "The number of classes must be a positive integer"
    end

    if Nx.rank(x.shape) != 2 do
      raise ArgumentError, "Training vector must be two-dimensional (n_samples, n_features)"
    end

    if (Nx.rank(y.shape) != 1 and !opts[:one_hot]) or (Nx.rank(y.shape) != 2 and opts[:one_hot]) do
      raise ArgumentError,
            "Target vector must be one-dimensional (n_samples) or two-dimensional if :one_hot set to true"
    end

    if (!is_nil(opts[:lr]) and !is_number(opts[:lr])) or (is_number(opts[:lr]) and opts[:lr] <= 0) do
      raise ArgumentError, "Learning rate must be a positive number"
    end

    if (!is_nil(opts[:iterations]) and !is_number(opts[:iterations])) or
         (is_integer(opts[:iterations]) and opts[:iterations] <= 0) do
      raise ArgumentError, "Number of iterations must be a positive integer"
    end
  end

  # Binary logistic regression

  defnp fit_binary(x, y, opts \\ []) do
    opts = keyword!(opts, iterations: 1000, lr: 0.01, num_classes: 2, one_hot: false)

    iterations = opts[:iterations]
    lr = opts[:lr]
    y = if opts[:one_hot], do: Nx.argmax(y, axis: 1), else: y
    x_t = Nx.transpose(x)
    y_t = Nx.transpose(y)

    {_m, n} = x.shape
    coeff = Nx.broadcast(Nx.tensor(0, type: {:f, 32}), {n})

    {_, _, _, _, _, _, final_coeff, final_bias} =
      while {iter = 0, x, lr, iterations, x_t, y_t, coeff, bias = Nx.tensor(0, type: {:f, 32})},
            Nx.less(iter, iterations) do
        {coeff, bias} = update_coefficients(x, x_t, y_t, {coeff, bias}, lr)
        {iter + 1, x, lr, iterations, x_t, y_t, coeff, bias}
      end

    %LogisticRegression{coefficients: final_coeff, bias: final_bias, mode: 0}
  end

  # Function computes one-hot encoding

  defnp one_hot_encoding(labels, num_classes) do
    Nx.equal(Nx.new_axis(labels, -1), Nx.iota({1, num_classes}))
  end

  # Multinomial logistic regression

  defnp fit_multinomial(x, y, opts) do
    {_m, n} = x.shape

    opts =
      keyword!(opts,
        iterations: 1000,
        lr: 0.01,
        num_classes: 1,
        one_hot: false
      )

    iterations = opts[:iterations]
    lr = opts[:lr]
    num_classes = opts[:num_classes]
    one_hot = if opts[:one_hot], do: y, else: one_hot_encoding(y, num_classes)
    x_t = Nx.transpose(x)

    {_, _, _, _, _, _,_, final_coeff} =
      while {iter = 0, x, lr, n, iterations, one_hot, x_t,
             coeff = Nx.broadcast(Nx.tensor(0, type: {:f, 32}), {n, num_classes})},
            Nx.less(iter, iterations) do
        coeff = update_coefficients_multinomial(x, x_t, one_hot, coeff, lr)
        {iter + 1, x, lr, n, iterations, one_hot, x_t, coeff}
      end

    %LogisticRegression{coefficients: final_coeff, bias: Nx.tensor(0, type: {:f, 32}), mode: 1}
  end

  # Normalized softmax

  defnp softmax(t) do
    normalized = t - Nx.reduce_max(t, axes: [1], keep_axes: true)
    Nx.transpose(Nx.transpose(Nx.exp(normalized)) / Nx.sum(Nx.exp(normalized), axes: [1]))
  end

  # Gradient descent for binary regression

  defnp update_coefficients(x, x_t, y_t, {coeff, bias}, lr) do
    {m, _n} = x.shape

    logit =
      x
      |> Nx.dot(coeff)
      |> Nx.add(bias)
      |> Nx.multiply(-1)
      |> Nx.exp()
      |> Nx.add(1)
      |> then(&(1 / &1))

    diff =
      logit
      |> Nx.subtract(y_t)
      |> Nx.reshape({m})

    coeff_diff =
      x_t
      |> Nx.dot(diff)
      |> Nx.divide(m)

    bias_diff =
      diff
      |> Nx.sum()
      |> Nx.divide(m)

    new_coeff = Nx.subtract(coeff, Nx.multiply(coeff_diff, lr))
    new_bias = Nx.subtract(bias, Nx.multiply(bias_diff, lr))

    {new_coeff, new_bias}
  end

  # Gradient descent for multinomial regression

  defnp update_coefficients_multinomial(x, x_t, one_hot, coeff, lr) do
    {m, _n} = x.shape

    dot_prod =
      x
      |> Nx.dot(coeff)

    prob = softmax(dot_prod)

    diff =
      prob
      |> Nx.subtract(one_hot)
      |> then(&Nx.dot(x_t, &1))
      |> Nx.divide(m)

    Nx.subtract(coeff, Nx.multiply(diff, lr))
  end

  @doc """
  Makes predictions with the given model on inputs `x`.
  """

  defn predict(%LogisticRegression{mode: mode} = model, x) do
    if mode == 0, do: predict_binary(model, x), else: predict_multinomial(model, x)
  end

  defnp predict_binary(%LogisticRegression{coefficients: coeff, bias: bias}, x) do
    logit =
      coeff
      |> Nx.dot(Nx.transpose(x))
      |> Nx.add(bias)
      |> Nx.multiply(-1)
      |> Nx.exp()
      |> Nx.add(1)
      |> then(&(1 / &1))

    logit > 0.5
  end

  defnp predict_multinomial(%LogisticRegression{coefficients: coeff}, x) do
    dot_prod = Nx.dot(x, coeff)
    prob = softmax(dot_prod)
    Nx.argmax(prob, axis: 1)
  end
end
