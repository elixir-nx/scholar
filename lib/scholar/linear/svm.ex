defmodule Scholar.Linear.SVM do
  @moduledoc """
  Support Vector Machine linear classifier.

  It uses the One-vs-Rest strategy to handle both binary and multinomial classification.
  This implementation uses stochastic gradient descent from default or any other optimizer
  available in `Polaris`. This makes it similar to a sklearn SGDClassifier [1].

  On average it is slower than algorithms that use QP and kernel trick (LIBSVM [2]) or
  Coordinate Descent Algorithm (LIBLINEAR [3]). It also cannot use different kernels
  like in LIBSVM, but you can use any type of optimizer available in `Polaris`.

  Time complexity is $O(N * K * I * C)$ where $N$ is the number of samples, $K$ is the
  number of features $I$ is the number of iterations and $C$ is the number of classes.

  [1] - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
  [2] - https://www.csie.ntu.edu.tw/~cjlin/libsvm/
  [3] - https://www.csie.ntu.edu.tw/~cjlin/liblinear/
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
      number of iterations of gradient descent performed inside SVM.
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
    ],
    loss_fn: [
      type: {:custom, Scholar.Linear.SVM, :loss_function, []},
      default: nil,
      doc: """
      The loss function that is used in the algorithm. Functions should take two arguments: `y_predicted` and `y_true`.
      If now provided it is set to highe loss without regularization.
      """
    ]
  ]

  def loss_function(function) do
    case function do
      function when is_function(function, 2) ->
        {:ok, function}

      nil ->
        loss_fn = &Scholar.Linear.SVM.hinge_loss(&1, &2, c: 1.0, margin: 10)

        {:ok, loss_fn}

      _ ->
        {:error,
         "expected loss function to be a function with arity 2, got: #{inspect(function)}"}
    end
  end

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits an SVM model for sample inputs `x` and sample
  targets `y`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:coefficients` - Coefficient of the features in the decision function.

    * `:bias` - Bias added to the decision function.

  ## Examples
      iex> x = Nx.tensor([[1.0, 2.0, 2.1], [3.0, 2.0, 1.4], [4.0, 7.0, 5.3], [3.0, 4.0, 6.3]])
      iex> y = Nx.tensor([1, 0, 1, 1])
      iex> Scholar.Linear.SVM.fit(x, y, num_classes: 2)
      %Scholar.Linear.SVM{
        coefficients: Nx.tensor(
          [
            [1.6899993419647217, 1.4599995613098145, 1.322001338005066],
            [1.4799995422363281, 1.9599990844726562, 2.0080013275146484]
          ]
        ),
        bias: Nx.tensor(
          [0.23000003397464752, 0.4799998104572296]
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
        {num_classes, n}
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

  # SVM training loop
  defnp fit_n(x, y, coef, bias, coef_optimizer_state, bias_optimizer_state, opts) do
    iterations = opts[:iterations]
    num_classes = opts[:num_classes]
    optimizer_update_fn = opts[:optimizer_update_fn]
    eps = opts[:eps]

    {{final_coef, final_bias}, _} =
      while {{coef, bias},
             {x, iterations, y, coef_optimizer_state, bias_optimizer_state,
              has_converged = Nx.broadcast(Nx.u8(0), {num_classes}), eps, iter = 0}},
            iter < iterations and not Nx.all(has_converged) do
        # ++++ inner while ++++++
        {{coef, bias, has_converged, coef_optimizer_state, bias_optimizer_state}, _} =
          while {{coef, bias, has_converged, coef_optimizer_state, bias_optimizer_state},
                 {x, y, iterations, iter, eps, j = 0}},
                j < num_classes do
            y_j = y == j
            coef_j = Nx.take(coef, j)
            bias_j = Nx.take(bias, j)

            {loss, {coef_grad, bias_grad}} = loss_and_grad(coef_j, bias_j, x, y_j, opts[:loss_fn])
            grad = Nx.broadcast(0.0, {num_classes, Nx.axis_size(x, 1)})
            coef_grad = Nx.put_slice(grad, [j, 0], Nx.new_axis(coef_grad, 0))

            {coef_updates, coef_optimizer_state} =
              optimizer_update_fn.(coef_grad, coef_optimizer_state, coef)

            coef = Polaris.Updates.apply_updates(coef, coef_updates)

            grad = Nx.broadcast(0.0, {num_classes})
            bias_grad = Nx.put_slice(grad, [j], Nx.new_axis(bias_grad, 0))

            {bias_updates, bias_optimizer_state} =
              optimizer_update_fn.(bias_grad, bias_optimizer_state, bias)

            bias = Polaris.Updates.apply_updates(bias, bias_updates)

            has_converged_j = Nx.sum(Nx.abs(loss)) < Nx.axis_size(x, 0) * eps

            has_converged =
              Nx.indexed_put(
                has_converged,
                Nx.new_axis(j, -1),
                has_converged_j
              )

            {{coef, bias, has_converged, coef_optimizer_state, bias_optimizer_state},
             {x, y, iterations, iter, eps, j + 1}}
          end

        # ++++ end inner while ++++++

        {{coef, bias},
         {x, iterations, y, coef_optimizer_state, bias_optimizer_state, has_converged, eps,
          iter + 1}}
      end

    %__MODULE__{
      coefficients: final_coef,
      bias: final_bias
    }
  end

  defnp loss_and_grad(coeff, bias, xs, ys, loss_fn) do
    value_and_grad({coeff, bias}, fn {coeff, bias} ->
      y_pred = predict(coeff, bias, xs)
      loss_fn.(y_pred, ys)
    end)
  end

  defnp predict(coeff, bias, xs) do
    Nx.dot(xs, [-1], coeff, [-1]) + bias
  end

  defn hinge_loss(y_pred, ys, opts \\ []) do
    c = opts[:c]
    margin = opts[:margin]
    c * Nx.sum(Nx.max(0, margin - y_pred) * ys, axes: [-1])
  end

  @doc """
  Makes predictions with the given model on inputs `x`.

  ## Examples
      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([1, 0, 1])
      iex> model = Scholar.Linear.SVM.fit(x, y, num_classes: 2)
      iex> Scholar.Linear.SVM.predict(model, Nx.tensor([[-3.0, 5.0]]))
      #Nx.Tensor<
        s64[1]
        [1]
      >
  """
  defn predict(%__MODULE__{coefficients: coeff, bias: bias}, x) do
    score = predict(coeff, bias, x)
    Nx.argmax(score, axis: -1)
  end
end
