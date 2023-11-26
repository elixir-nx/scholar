defmodule Scholar.Linear.LogisticRegression do
  @moduledoc """
  Logistic regression in both binary and multinomial variants.

  Time complexity is $O(N * K * I)$ where $N$ is the number of samples, $K$ is the number of features, and $I$ is the number of iterations.
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, containers: [:coefficients, :bias]}
  defstruct [:coefficients, :bias]

  opts = [
    num_classes: [
      required: true,
      type: :pos_integer,
      doc: "number of classes contained in the input tensors."
    ],
    iterations: [
      type: :pos_integer,
      default: 1000,
      doc: """
      number of iterations of gradient descent performed inside logistic
      regression.
      """
    ],
    learning_loop_unroll: [
      type: :boolean,
      default: false,
      doc: ~S"""
      If `true`, the learning loop is unrolled.
      """
    ],
    optimizer: [
      type: {:custom, Scholar.Options, :optimizer, []},
      default: :sgd,
      doc: """
      The optimizer name or {init, update} pair of functions (see `Polaris.Optimizers` for more details).
      """
    ],
    eps: [
      type: :float,
      default: 1.0e-8,
      doc:
        "The convergence tolerance. If the `abs(loss) < size(x) * :eps`, the algorithm is considered to have converged."
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
            [2.5531527996063232, -0.5531544089317322],
            [-0.35652396082878113, 2.3565237522125244]
          ]
        ),
        bias: Nx.tensor(
          [-0.28847914934158325, 0.28847917914390564]
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

    opts = NimbleOptions.validate!(opts, @opts_schema)

    {optimizer, opts} = Keyword.pop!(opts, :optimizer)

    {optimizer_init_fn, optimizer_update_fn} =
      case optimizer do
        atom when is_atom(atom) -> apply(Polaris.Optimizers, atom, [])
        {f1, f2} -> {f1, f2}
      end

    n = Nx.axis_size(x, -1)
    num_classes = opts[:num_classes]

    coef =
      Nx.broadcast(
        Nx.tensor(1.0, type: to_float_type(x)),
        {n, num_classes}
      )

    bias = Nx.broadcast(Nx.tensor(0, type: to_float_type(x)), {num_classes})

    coef_optimizer_state = optimizer_init_fn.(coef) |> as_type(to_float_type(x))
    bias_optimizer_state = optimizer_init_fn.(bias) |> as_type(to_float_type(x))

    opts = Keyword.put(opts, :optimizer_update_fn, optimizer_update_fn)

    fit_n(x, y, coef, bias, coef_optimizer_state, bias_optimizer_state, opts)
  end

  deftransformp as_type(container, target_type) do
    Nx.Defn.Composite.traverse(container, fn t ->
      type = Nx.type(t)

      if Nx.Type.float?(type) and not Nx.Type.complex?(type) do
        Nx.as_type(t, target_type)
      else
        t
      end
    end)
  end

  # Logistic Regression training loop

  defnp fit_n(x, y, coef, bias, coef_optimizer_state, bias_optimizer_state, opts) do
    iterations = opts[:iterations]
    num_classes = opts[:num_classes]
    optimizer_update_fn = opts[:optimizer_update_fn]
    y = Scholar.Preprocessing.one_hot_encode(y, num_classes: num_classes)

    {{final_coef, final_bias}, _} =
      while {{coef, bias},
             {x, iterations, y, coef_optimizer_state, bias_optimizer_state,
              has_converged = Nx.u8(0), iter = 0}},
            iter < iterations and not has_converged do
        {loss, {coef_grad, bias_grad}} = loss_and_grad(coef, bias, x, y)

        {coef_updates, coef_optimizer_state} =
          optimizer_update_fn.(coef_grad, coef_optimizer_state, coef)

        coef = Polaris.Updates.apply_updates(coef, coef_updates)

        {bias_updates, bias_optimizer_state} =
          optimizer_update_fn.(bias_grad, bias_optimizer_state, bias)

        bias = Polaris.Updates.apply_updates(bias, bias_updates)

        has_converged = Nx.sum(Nx.abs(loss)) < Nx.size(x) * opts[:eps]

        {{coef, bias},
         {x, iterations, y, coef_optimizer_state, bias_optimizer_state, has_converged, iter + 1}}
      end

    %__MODULE__{
      coefficients: final_coef,
      bias: final_bias
    }
  end

  defnp loss_and_grad(coeff, bias, xs, ys) do
    value_and_grad({coeff, bias}, fn {coeff, bias} ->
      -Nx.sum(ys * log_softmax(Nx.dot(xs, coeff) + bias), axes: [-1])
    end)
  end

  defnp log_softmax(x) do
    shifted = x - stop_grad(Nx.reduce_max(x, axes: [-1], keep_axes: true))

    shifted
    |> Nx.exp()
    |> Nx.sum(axes: [-1], keep_axes: true)
    |> Nx.log()
    |> Nx.negate()
    |> Nx.add(shifted)
  end

  # Normalized softmax

  defnp softmax(t) do
    max = stop_grad(Nx.reduce_max(t, axes: [-1], keep_axes: true))
    normalized_exp = (t - max) |> Nx.exp()
    normalized_exp / Nx.sum(normalized_exp, axes: [-1], keep_axes: true)
  end

  @doc """
  Makes predictions with the given `model` on inputs `x`.

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
  defn predict(%__MODULE__{coefficients: coeff, bias: bias} = _model, x) do
    inter = Nx.dot(x, [1], coeff, [0]) + bias
    Nx.argmax(inter, axis: 1)
  end

  @doc """
  Calculates probabilities of predictions with the given `model` on inputs `x`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([1, 0, 1])
      iex> model = Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2)
      iex> Scholar.Linear.LogisticRegression.predict_probability(model, Nx.tensor([[-3.0, 5.0]]))
      #Nx.Tensor<
        f32[1][2]
        [
          [6.470913388456623e-11, 1.0]
        ]
      >
  """
  defn predict_probability(%__MODULE__{coefficients: coeff, bias: bias} = _model, x) do
    softmax(Nx.dot(x, [1], coeff, [0]) + bias)
  end
end
