defmodule Scholar.Linear.LogisticRegression do
  @moduledoc """
  Multiclass logistic regression.

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
      doc: "Number of output classes."
    ],
    max_iterations: [
      type: :pos_integer,
      default: 1000,
      doc: "Maximum number of gradient descent iterations to perform."
    ],
    optimizer: [
      type: {:custom, Scholar.Options, :optimizer, []},
      default: :sgd,
      doc: """
      Optimizer name or {init, update} pair of functions (see `Polaris.Optimizers` for more details).
      """
    ],
    alpha: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0,
      doc: """
      Constant that multiplies the regularization term, controlling regularization strength.
      If 0, no regularization is applied.
      """
    ],
    l1_ratio: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 0.0,
      doc: """
      The Elastic-Net mixing parameter, with `0 <= l1_ratio <= 1`.
      Setting `l1_ratio` to 0 gives pure L2 regularization, and setting it to 1 gives pure L1 regularization.
      For values between 0 and 1, a penalty of the form `l1_ratio * L1 + (1 - l1_ratio) * L2` is used.
      """
    ],
    tol: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0e-4,
      doc: """
      Convergence tolerance. If the infinity norm of the gradient is less than `:tol`,
      the algorithm is considered to have converged.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a logistic regression model for sample inputs `x` and sample
  targets `y`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:coefficients` - Coefficient of the features in the decision function.

    * `:bias` - Bias added to the decision function.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([1, 0, 1])
      iex> Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2)
      %Scholar.Linear.LogisticRegression{
        coefficients: Nx.tensor(
          [
            [0.09002052247524261, -0.09002052992582321],
            [-0.1521512120962143, 0.1521512120962143]
          ]
        ),
        bias: Nx.tensor([-0.05300388112664223, 0.053003907203674316])
      }
  """
  deftransform fit(x, y, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {num_samples, num_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    if Nx.rank(y) != 1 do
      raise ArgumentError,
            "expected y to have shape {num_samples}, got tensor with shape: #{inspect(Nx.shape(y))}"
    end

    {num_samples, num_features} = Nx.shape(x)

    if Nx.axis_size(y, 0) != num_samples do
      raise ArgumentError,
            "expected x and y to have the same number of samples, got #{num_samples} and #{Nx.axis_size(y, 0)}"
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)

    {l1_ratio, opts} = Keyword.pop!(opts, :l1_ratio)

    unless l1_ratio >= 0.0 and l1_ratio <= 1.0 do
      raise ArgumentError,
            "expected l1_ratio to be between 0 and 1, got: #{inspect(l1_ratio)}"
    end

    type = to_float_type(x)
    {optimizer, opts} = Keyword.pop!(opts, :optimizer)

    {optimizer_init_fn, optimizer_update_fn} =
      case optimizer do
        atom when is_atom(atom) -> apply(Polaris.Optimizers, atom, [])
        {f1, f2} -> {f1, f2}
      end

    num_classes = opts[:num_classes]

    coef =
      Nx.broadcast(
        Nx.tensor(0.0, type: type),
        {num_features, num_classes}
      )

    bias = Nx.broadcast(Nx.tensor(0.0, type: type), {num_classes})

    coef_optimizer_state = optimizer_init_fn.(coef) |> as_type(type)
    bias_optimizer_state = optimizer_init_fn.(bias) |> as_type(type)

    {alpha, opts} = Keyword.pop!(opts, :alpha)
    {tol, opts} = Keyword.pop!(opts, :tol)
    alpha = Nx.tensor(alpha, type: type)
    l1_ratio = Nx.tensor(l1_ratio, type: type)
    tol = Nx.tensor(tol, type: type)

    opts = Keyword.put(opts, :optimizer_update_fn, optimizer_update_fn)

    fit_n(
      x,
      y,
      coef,
      bias,
      alpha,
      l1_ratio,
      tol,
      coef_optimizer_state,
      bias_optimizer_state,
      opts
    )
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

  defnp fit_n(
          x,
          y,
          coef,
          bias,
          alpha,
          l1_ratio,
          tol,
          coef_optimizer_state,
          bias_optimizer_state,
          opts
        ) do
    num_samples = Nx.axis_size(x, 0)
    max_iterations = opts[:max_iterations]
    num_classes = opts[:num_classes]
    optimizer_update_fn = opts[:optimizer_update_fn]

    y_one_hot =
      y
      |> Nx.new_axis(1)
      |> Nx.broadcast({num_samples, num_classes})
      |> Nx.equal(Nx.iota({num_samples, num_classes}, axis: 1))

    {final_coef, final_bias, _} =
      while {coef, bias,
             {x, y_one_hot, max_iterations, alpha, l1_ratio, tol, coef_optimizer_state,
              bias_optimizer_state, converged? = Nx.u8(0), iter = Nx.u32(0)}},
            iter < max_iterations and not converged? do
        {coef_grad, bias_grad} =
          grad({coef, bias}, fn {coef, bias} ->
            compute_loss(coef, bias, alpha, l1_ratio, x, y_one_hot)
          end)

        {coef_updates, coef_optimizer_state} =
          optimizer_update_fn.(coef_grad, coef_optimizer_state, coef)

        coef = Polaris.Updates.apply_updates(coef, coef_updates)

        {bias_updates, bias_optimizer_state} =
          optimizer_update_fn.(bias_grad, bias_optimizer_state, bias)

        bias = Polaris.Updates.apply_updates(bias, bias_updates)

        converged? =
          Nx.reduce_max(Nx.abs(coef_grad)) < tol and Nx.reduce_max(Nx.abs(bias_grad)) < tol

        {coef, bias,
         {x, y_one_hot, max_iterations, alpha, l1_ratio, tol, coef_optimizer_state,
          bias_optimizer_state, converged?, iter + 1}}
      end

    %__MODULE__{
      coefficients: final_coef,
      bias: final_bias
    }
  end

  defnp compute_regularization(coeff, alpha, l1_ratio) do
    if alpha > 0.0 do
      reg =
        cond do
          l1_ratio == 0.0 ->
            # L2 regularization
            Nx.sum(coeff * coeff)

          l1_ratio == 1.0 ->
            # L1 regularization
            Nx.sum(Nx.abs(coeff))

          # Elastic-Net regularization
          true ->
            l1_ratio * Nx.sum(Nx.abs(coeff)) +
              (1 - l1_ratio) * Nx.sum(coeff * coeff)
        end

      alpha * reg
    else
      0.0
    end
  end

  defnp compute_loss(coeff, bias, alpha, l1_ratio, xs, ys) do
    reg = compute_regularization(coeff, alpha, l1_ratio)

    xs
    |> Nx.dot(coeff)
    |> Nx.add(bias)
    |> log_softmax()
    |> Nx.multiply(ys)
    |> Nx.sum(axes: [1])
    |> Nx.negate()
    |> Nx.mean()
    |> Nx.add(reg)
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

  Output predictions have shape `{n_samples}` when train target is shaped either `{n_samples}` or `{n_samples, 1}`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([1, 0, 1])
      iex> model = Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2)
      iex> Scholar.Linear.LogisticRegression.predict(model, Nx.tensor([[-3.0, 5.0]]))
      Nx.tensor([1])
  """
  defn predict(%__MODULE__{coefficients: coeff, bias: bias} = _model, x) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    logits = Nx.dot(x, coeff) + bias
    Nx.argmax(logits, axis: 1)
  end

  @doc """
  Calculates probabilities of predictions with the given `model` on inputs `x`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([1, 0, 1])
      iex> model = Scholar.Linear.LogisticRegression.fit(x, y, num_classes: 2)
      iex> Scholar.Linear.LogisticRegression.predict_probability(model, Nx.tensor([[-3.0, 5.0]]))
      Nx.tensor([[0.10269401967525482, 0.8973060250282288]])
  """
  defn predict_probability(%__MODULE__{coefficients: coeff, bias: bias} = _model, x) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    softmax(Nx.dot(x, coeff) + bias)
  end
end
