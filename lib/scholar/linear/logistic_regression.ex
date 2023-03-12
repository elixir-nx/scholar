defmodule Scholar.Linear.LogisticRegression do
  @moduledoc """
  Logistic regression in both binary and multinomial variants.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :bias]}
  defstruct [:coefficients, :bias]

  opts = [
    num_classes: [
      required: true,
      type: :pos_integer,
      doc: "number of classes contained in the input tensors."
    ],
    learning_rate: [
      type: {:custom, Scholar.Options, :positive_number, []},
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

  ## Return Values

    The function returns a struct with the following parameters:

    * `:coefficients` - Coefficient of the features in the decision function.

    * `:bias` - Bias added to the decision function.

    * `:mode` - Indicates whether the problem is binary classification (`:num_classes` set to 2)
      or multinomial (`:num_classes` is bigger than 2). For binary classification set to `:binary`, otherwise
      set to `:multinomial`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([1, 0, 1])
      iex> Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2)
      %Scholar.Linear.LogisticRegression{
        coefficients: Nx.tensor(
          [
            [2.0225090980529785, 0.0742921307682991],
            [-0.02250923588871956, 1.9257065057754517]
          ]
        ),
        bias: Nx.tensor(
          [-0.11229748278856277, 0.11229748278856277]
        )
      }
  """
  deftransform fit(x, y, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    if Nx.rank(y) != 1 do
      raise ArgumentError,
            "expected y to have shape {n_samples}, got tensor with shape: #{inspect(Nx.shape(y))}"
    end

    fit_n(x, y, NimbleOptions.validate!(opts, @opts_schema))
  end

  # Logistic Regression training loop

  defnp fit_n(x, y, opts) do
    {_m, n} = x.shape
    iterations = opts[:iterations]
    learning_rate = opts[:learning_rate]
    num_classes = opts[:num_classes]
    y = Scholar.Preprocessing.one_hot_encode(y, num_classes: num_classes)

    {_, _, _, _, _, _, final_coeff, final_bias} =
      while {iter = 0, x, learning_rate, n, iterations, y,
             coeff =
               Nx.broadcast(
                 Nx.tensor(1.0, type: Nx.Type.to_floating(Nx.type(x))),
                 {n, num_classes}
               ),
             bias =
               Nx.broadcast(Nx.tensor(0, type: Nx.Type.to_floating(Nx.type(x))), {num_classes})},
            iter < iterations do
        {coeff_grad, bias_grad} = grad_loss(coeff, bias, x, y)
        coeff = coeff - learning_rate * coeff_grad
        bias = bias - learning_rate * bias_grad
        {iter + 1, x, learning_rate, n, iterations, y, coeff, bias}
      end

    %__MODULE__{
      coefficients: Nx.transpose(final_coeff),
      bias: final_bias
    }
  end

  defnp cross_entropy(log_probs, targets) do
    target_class = Nx.argmax(targets, axis: 1)
    nll = Nx.take_along_axis(log_probs, Nx.new_axis(target_class, 1), axis: 1)
    -Nx.mean(nll)
  end

  defnp loss_fn(coeff, bias, xs, ys) do
    probs = Axon.Activations.log_softmax(Nx.dot(xs, coeff) + bias)
    cross_entropy(probs, ys)
  end

  defnp grad_loss(coeff, bias, xs, ys) do
    grad({coeff, bias}, fn {coeff, bias} -> loss_fn(coeff, bias, xs, ys) end)
  end

  # Normalized softmax

  defn softmax(t) do
    max = Nx.Defn.Kernel.stop_grad(Nx.reduce_max(t, axes: [-1], keep_axes: true))
    normalized_exp = (t - max) |> Nx.exp()
    normalized_exp / Nx.sum(normalized_exp, axes: [-1], keep_axes: true)
  end

  @doc """
  Makes predictions with the given model on inputs `x`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([1, 0, 1])
      iex> model = Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2)
      iex> Scholar.Linear.LogisticRegression.predict(model, Nx.tensor([[-3.0, 5.0]]))
      #Nx.Tensor<
        s64[1]
        [1]
      >
  """
  defn predict(%__MODULE__{coefficients: coeff, bias: bias}, x) do
    inter = Nx.dot(x, Nx.transpose(coeff)) + bias
    Nx.argmax(inter, axis: 1)
  end

  @doc """
  Calculates probabilities of predictions with the given model on inputs `x`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([1, 0, 1])
      iex> model = Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2)
      iex> Scholar.Linear.LogisticRegression.predict_probability(model, Nx.tensor([[-3.0, 5.0]]))
      #Nx.Tensor<
        f32[1][2]
        [
          [1.650987258017267e-7, 0.9999998807907104]
        ]
      >
  """
  defn predict_probability(%__MODULE__{coefficients: coeff, bias: bias}, x) do
    softmax(Nx.dot(x, Nx.transpose(coeff)) + bias)
  end
end
