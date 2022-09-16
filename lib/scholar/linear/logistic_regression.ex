defmodule Scholar.Linear.LogisticRegression do
  @moduledoc """
  Logistic regression in both binary and multinomial variants.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :bias], keep: [:mode]}
  defstruct [:coefficients, :bias, :mode]

  opts = [
    num_classes: [
      required: true,
      type: :pos_integer,
      doc: "number of classes contained in the input tensors."
    ],
    learning_rate: [
      type: {:custom, Scholar.Shared, :check_if_positive_float, [:learning_rate]},
      default: 0.01,
      doc: """
      learning rate used by gradient descent.
      """
    ],
    iterations: [
      type: :pos_integer,
      default: 1000,
      doc: """
      number of iterations of gradient descent performed inside logistic
      regression.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a logistic regression model for sample inputs `x` and sample
  targets `y`.

  Depending on number of classes the function chooses either binary
  or multinomial logistic regression.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  """
  deftransform train(x, y, opts \\ []) do
    ntrain(x, y, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp ntrain(x, y, opts \\ []) do
    fit_verify(x, y)

    if opts[:num_classes] < 3 do
      fit_binary(x, y, opts)
    else
      fit_multinomial(x, y, opts)
    end
  end

  # Function checks validity of the provided data

  deftransformp fit_verify(x, y) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    if Nx.rank(y) != 1 do
      raise ArgumentError,
            "expected y to have shape {n_samples}, got tensor with shape: #{inspect(Nx.shape(y))}"
    end
  end

  # Binary logistic regression

  defnp fit_binary(x, y, opts \\ []) do
    iterations = opts[:iterations]
    learning_rate = opts[:learning_rate]
    x_t = Nx.transpose(x)
    y_t = Nx.transpose(y)

    {_m, n} = Nx.shape(x)
    coeff = Nx.broadcast(Nx.tensor(0, type: {:f, 32}), {n})

    {_, _, _, _, _, _, final_coeff, final_bias} =
      while {iter = 0, x, learning_rate, iterations, x_t, y_t, coeff,
             bias = Nx.tensor(0, type: {:f, 32})},
            Nx.less(iter, iterations) do
        {coeff, bias} = update_coefficients(x, x_t, y_t, {coeff, bias}, learning_rate)
        {iter + 1, x, learning_rate, iterations, x_t, y_t, coeff, bias}
      end

    %__MODULE__{coefficients: final_coeff, bias: final_bias, mode: :binary}
  end

  # Function computes one-hot encoding

  defnp one_hot_encoding(labels, num_classes) do
    Nx.equal(Nx.new_axis(labels, -1), Nx.iota({1, num_classes}))
  end

  # Multinomial logistic regression

  defnp fit_multinomial(x, y, opts) do
    {_m, n} = x.shape
    iterations = opts[:iterations]
    learning_rate = opts[:learning_rate]
    num_classes = opts[:num_classes]
    one_hot = one_hot_encoding(y, num_classes)
    x_t = Nx.transpose(x)

    {_, _, _, _, _, _, _, final_coeff} =
      while {iter = 0, x, learning_rate, n, iterations, one_hot, x_t,
             coeff = Nx.broadcast(Nx.tensor(0, type: {:f, 32}), {n, num_classes})},
            iter < iterations do
        coeff = update_coefficients_multinomial(x, x_t, one_hot, coeff, learning_rate)
        {iter + 1, x, learning_rate, n, iterations, one_hot, x_t, coeff}
      end

    %__MODULE__{coefficients: final_coeff, bias: Nx.tensor(0, type: {:f, 32}), mode: :multinomial}
  end

  # Normalized softmax

  defnp softmax(t) do
    normalized_exp = (t - Nx.reduce_max(t, axes: [0], keep_axes: true)) |> Nx.exp()
    normalized_exp / Nx.sum(normalized_exp, axes: [0])
  end

  # Gradient descent for binary regression

  defnp update_coefficients(x, x_t, y_t, {coeff, bias}, learning_rate) do
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
      (logit - y_t)
      |> Nx.reshape({m})

    coeff_diff =
      x_t
      |> Nx.dot(diff)
      |> Nx.divide(m)

    bias_diff =
      diff
      |> Nx.sum()
      |> Nx.divide(m)

    new_coeff = coeff - coeff_diff * learning_rate
    new_bias = bias - bias_diff * learning_rate

    {new_coeff, new_bias}
  end

  # Gradient descent for multinomial regression

  defnp update_coefficients_multinomial(x, x_t, one_hot, coeff, learning_rate) do
    {m, _n} = x.shape

    dot_prod =
      x
      |> Nx.dot(coeff)

    prob = softmax(dot_prod)

    diff = Nx.dot(x_t, prob - one_hot) / m

    coeff - diff * learning_rate
  end

  @doc """
  Makes predictions with the given model on inputs `x`.
  """

  defn predict(%__MODULE__{mode: mode} = model, x) do
    case mode do
      :binary -> predict_binary(model, x)
      :multinomial -> predict_multinomial(model, x)
    end
  end

  defnp predict_binary(%__MODULE__{coefficients: coeff, bias: bias}, x) do
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

  defnp predict_multinomial(%__MODULE__{coefficients: coeff}, x) do
    dot_prod = Nx.dot(x, coeff)
    Nx.argmax(dot_prod, axis: 1)
  end
end
