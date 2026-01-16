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
    alpha: [
      type: {:custom, Scholar.Options, :non_negative_number, []},
      default: 1.0,
      doc: """
      Constant that multiplies the L2 regularization term, controlling regularization strength.
      If 0, no regularization is applied.
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
            [0.0915902629494667, -0.09159023314714432],
            [-0.1507941037416458, 0.1507941335439682]
          ]
        ),
        bias: Nx.tensor([-0.06566660106182098, 0.06566664576530457])
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

    num_samples = Nx.axis_size(x, 0)

    if Nx.axis_size(y, 0) != num_samples do
      raise ArgumentError,
            "expected x and y to have the same number of samples, got #{num_samples} and #{Nx.axis_size(y, 0)}"
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)

    type = to_float_type(x)

    {alpha, opts} = Keyword.pop!(opts, :alpha)
    alpha = Nx.tensor(alpha, type: type)
    {tol, opts} = Keyword.pop!(opts, :tol)
    tol = Nx.tensor(tol, type: type)

    fit_n(x, y, alpha, tol, opts)
  end

  defnp fit_n(x, y, alpha, tol, opts) do
    num_classes = opts[:num_classes]
    max_iterations = opts[:max_iterations]
    {num_samples, num_features} = Nx.shape(x)

    type = to_float_type(x)

    # Initialize weights and bias with zeros
    w =
      Nx.broadcast(
        Nx.tensor(0.0, type: type),
        {num_features, num_classes}
      )

    b = Nx.broadcast(Nx.tensor(0.0, type: type), {num_classes})

    # One-hot encoding of target labels
    y_one_hot =
      y
      |> Nx.new_axis(1)
      |> Nx.broadcast({num_samples, num_classes})
      |> Nx.equal(Nx.iota({num_samples, num_classes}, axis: 1))

    # Define Armijo parameters
    c = Nx.tensor(1.0e-4, type: type)
    rho = Nx.tensor(0.5, type: type)

    eta_min =
      case type do
        {:f, 32} -> Nx.tensor(1.0e-6, type: type)
        {:f, 64} -> Nx.tensor(1.0e-8, type: type)
        _ -> Nx.tensor(1.0e-6, type: type)
      end

    armijo_params = %{
      c: c,
      rho: rho,
      eta_min: eta_min
    }

    {coef, bias, _} =
      while {w, b,
             {alpha, x, y_one_hot, tol, armijo_params, iter = Nx.u32(0), converged? = Nx.u8(0)}},
            iter < max_iterations and not converged? do
        logits = Nx.dot(x, w) + b
        probabilities = softmax(logits)
        residuals = probabilities - y_one_hot

        # Compute loss
        loss =
          logits
          |> log_softmax()
          |> Nx.multiply(y_one_hot)
          |> Nx.sum(axes: [1])
          |> Nx.mean()
          |> Nx.negate()
          |> Nx.add(alpha * Nx.sum(w * w))

        # Compute gradients
        grad_w = Nx.dot(x, [0], residuals, [0]) / num_samples + 2 * alpha * w
        grad_b = Nx.sum(residuals, axes: [0]) / num_samples

        # Perform line search to find step size
        eta =
          armijo_line_search(w, b, alpha, x, y_one_hot, loss, grad_w, grad_b, armijo_params)

        w = w - eta * grad_w
        b = b - eta * grad_b

        converged? =
          Nx.reduce_max(Nx.abs(grad_w)) < tol and Nx.reduce_max(Nx.abs(grad_b)) < tol

        {w, b, {alpha, x, y_one_hot, tol, armijo_params, iter + 1, converged?}}
      end

    %__MODULE__{
      coefficients: coef,
      bias: bias
    }
  end

  defnp armijo_line_search(w, b, alpha, x, y, loss, grad_w, grad_b, armijo_params) do
    c = armijo_params[:c]
    rho = armijo_params[:rho]
    eta_min = armijo_params[:eta_min]

    type = to_float_type(x)
    dir_w = -grad_w
    dir_b = -grad_b
    # Directional derivative
    slope = Nx.sum(dir_w * grad_w) + Nx.sum(dir_b * grad_b)

    {eta, _} =
      while {eta = Nx.tensor(1.0, type: type),
             {w, b, alpha, x, y, loss, dir_w, dir_b, slope, c, rho, eta_min}},
            compute_loss(w + eta * dir_w, b + eta * dir_b, alpha, x, y) > loss + c * eta * slope and
              eta > eta_min do
        eta = eta * rho

        {eta, {w, b, alpha, x, y, loss, dir_w, dir_b, slope, c, rho, eta_min}}
      end

    eta
  end

  defnp compute_loss(w, b, alpha, x, y) do
    x
    |> Nx.dot(w)
    |> Nx.add(b)
    |> log_softmax()
    |> Nx.multiply(y)
    |> Nx.sum(axes: [1])
    |> Nx.mean()
    |> Nx.negate()
    |> Nx.add(alpha * Nx.sum(w * w))
  end

  defnp softmax(logits) do
    max = stop_grad(Nx.reduce_max(logits, axes: [1], keep_axes: true))
    normalized_exp = (logits - max) |> Nx.exp()
    normalized_exp / Nx.sum(normalized_exp, axes: [1], keep_axes: true)
  end

  defnp log_softmax(x) do
    shifted = x - stop_grad(Nx.reduce_max(x, axes: [1], keep_axes: true))

    shifted
    |> Nx.exp()
    |> Nx.sum(axes: [1], keep_axes: true)
    |> Nx.log()
    |> Nx.negate()
    |> Nx.add(shifted)
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
      Nx.tensor([[0.10075931251049042, 0.8992406725883484]])
  """
  defn predict_probability(%__MODULE__{coefficients: coeff, bias: bias} = _model, x) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    softmax(Nx.dot(x, coeff) + bias)
  end
end
