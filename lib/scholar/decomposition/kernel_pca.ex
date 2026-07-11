defmodule Scholar.Decomposition.KernelPCA do
  @moduledoc """
  Kernel Principal Component Analysis (Kernel PCA).

  Kernel PCA is an extension of `Scholar.Decomposition.PCA` that performs the
  decomposition in a reproducing kernel Hilbert space instead of the original
  feature space. Replacing the inner product with a kernel function lets the
  method capture non-linear structure that ordinary PCA cannot.

  The time complexity is $O(N^3)$ where $N$ is the number of samples, since it
  relies on the eigendecomposition of the $N \\times N$ kernel matrix.

  References:

  * [1] Schölkopf, B., Smola, A., & Müller, K. R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. Neural computation, 10(5), 1299-1319.
  """
  import Nx.Defn

  @derive {Nx.Container,
           keep: [:kernel, :gamma, :degree, :coef0],
           containers: [:eigenvalues, :eigenvectors, :x_fit, :kernel_fit_rows, :kernel_fit_all]}
  defstruct [
    :eigenvalues,
    :eigenvectors,
    :x_fit,
    :kernel_fit_rows,
    :kernel_fit_all,
    :kernel,
    :gamma,
    :degree,
    :coef0
  ]

  opts = [
    num_components: [
      required: true,
      type: :pos_integer,
      doc: "The number of principal components to keep."
    ],
    kernel: [
      type: {:in, [:linear, :poly, :rbf, :sigmoid, :cosine]},
      default: :linear,
      doc: "The kernel used to compute the pairwise similarities between samples."
    ],
    gamma: [
      type: {:or, [:float, nil]},
      default: nil,
      doc: """
      Kernel coefficient for the `:rbf`, `:poly` and `:sigmoid` kernels.
      When `nil` it defaults to `1 / num_features`.
      """
    ],
    degree: [
      type: :pos_integer,
      default: 3,
      doc: "Degree of the `:poly` kernel."
    ],
    coef0: [
      type: :float,
      default: 1.0,
      doc: "Independent term of the `:poly` and `:sigmoid` kernels."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a Kernel PCA for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:eigenvalues` - The eigenvalues of the centered kernel matrix, sorted in
      decreasing order.

    * `:eigenvectors` - The eigenvectors of the centered kernel matrix.

    * `:x_fit` - The training data, kept to compute the kernel against new samples.

    * `:kernel_fit_rows` - Column means of the training kernel matrix, used to
      center the kernel of new samples.

    * `:kernel_fit_all` - Mean of the whole training kernel matrix.

  ## Examples

      iex> x = Nx.tensor([[0.5, 0.2, 0.8], [1.0, 0.5, 0.2], [0.3, 1.0, 0.7], [0.9, 0.1, 1.0]])
      iex> kpca = Scholar.Decomposition.KernelPCA.fit(x, num_components: 2, kernel: :rbf)
      iex> kpca.eigenvalues
      Nx.tensor([0.3644832372665405, 0.26074573397636414])
  """
  deftransform fit(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}\
            """
    end

    {num_samples, num_features} = Nx.shape(x)
    num_components = opts[:num_components]

    if num_components > num_samples do
      raise ArgumentError,
            """
            num_components must be less than or equal to \
            num_samples = #{num_samples}, got #{num_components}\
            """
    end

    opts = Keyword.put(opts, :gamma, opts[:gamma] || 1.0 / num_features)
    fit_n(x, opts)
  end

  defnp fit_n(x, opts) do
    kernel = kernel_matrix(x, x, opts)
    {kernel_centered, kernel_fit_rows, kernel_fit_all} = center_fit(kernel)

    {eigenvalues, eigenvectors} = top_eigen(kernel_centered, opts[:num_components])

    %__MODULE__{
      eigenvalues: eigenvalues,
      eigenvectors: eigenvectors,
      x_fit: x,
      kernel_fit_rows: kernel_fit_rows,
      kernel_fit_all: kernel_fit_all,
      kernel: opts[:kernel],
      gamma: opts[:gamma],
      degree: opts[:degree],
      coef0: opts[:coef0]
    }
  end

  @doc """
  For a fitted `model` projects samples `x` onto the principal components.

  ## Return Values

  The function returns a tensor with the decomposed data.

  ## Examples

      iex> x = Nx.tensor([[0.5, 0.2, 0.8], [1.0, 0.5, 0.2], [0.3, 1.0, 0.7], [0.9, 0.1, 1.0]])
      iex> kpca = Scholar.Decomposition.KernelPCA.fit(x, num_components: 2, kernel: :rbf)
      iex> Scholar.Decomposition.KernelPCA.transform(kpca, Nx.tensor([[0.5, 0.5, 0.5]]))
      Nx.tensor(
        [
          [0.12500189244747162, 0.029097510501742363]
        ]
      )
  """
  deftransform transform(%__MODULE__{} = model, x) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}\
            """
    end

    transform_n(model, x)
  end

  defnp transform_n(%__MODULE__{x_fit: x_fit} = model, x) do
    kernel = kernel_matrix(x, x_fit, model_opts(model))
    kernel_centered = center_transform(kernel, model.kernel_fit_rows, model.kernel_fit_all)
    project(kernel_centered, model.eigenvalues, model.eigenvectors)
  end

  @doc """
  Fits a Kernel PCA on `x` and projects `x` onto the principal components.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a tensor with the decomposed data.

  ## Examples

      iex> x = Nx.tensor([[0.5, 0.2, 0.8], [1.0, 0.5, 0.2], [0.3, 1.0, 0.7], [0.9, 0.1, 1.0]])
      iex> Scholar.Decomposition.KernelPCA.fit_transform(x, num_components: 2, kernel: :rbf)
      Nx.tensor(
        [
          [-0.13561560213565826, -0.16519638895988464],
          [0.021600963547825813, 0.44114428758621216],
          [0.4687873125076294, -0.15764620900154114],
          [-0.35477250814437866, -0.11830171197652817]
        ]
      )
  """
  deftransform fit_transform(x, opts \\ []) do
    fit(x, opts) |> project_fit()
  end

  defnp project_fit(model) do
    model.eigenvectors * Nx.sqrt(model.eigenvalues)
  end

  # Kernel matrix between `x` (m samples) and `y` (n samples), shape {m, n}.
  defnp kernel_matrix(x, y, opts) do
    case opts[:kernel] do
      :linear ->
        Nx.dot(x, [1], y, [1])

      :poly ->
        (opts[:gamma] * Nx.dot(x, [1], y, [1]) + opts[:coef0]) ** opts[:degree]

      :rbf ->
        Nx.exp(-opts[:gamma] * Scholar.Metrics.Distance.pairwise_squared_euclidean(x, y))

      :sigmoid ->
        Nx.tanh(opts[:gamma] * Nx.dot(x, [1], y, [1]) + opts[:coef0])

      :cosine ->
        x_normalized = x / Nx.sqrt(Nx.sum(x * x, axes: [1], keep_axes: true))
        y_normalized = y / Nx.sqrt(Nx.sum(y * y, axes: [1], keep_axes: true))
        Nx.dot(x_normalized, [1], y_normalized, [1])
    end
  end

  deftransformp model_opts(model) do
    [kernel: model.kernel, gamma: model.gamma, degree: model.degree, coef0: model.coef0]
  end

  # Double centering of the training kernel, plus the statistics reused at transform time.
  defnp center_fit(kernel) do
    column_means = Nx.mean(kernel, axes: [0], keep_axes: true)
    row_means = Nx.mean(kernel, axes: [1], keep_axes: true)
    all_mean = Nx.mean(kernel)
    centered = kernel - column_means - row_means + all_mean
    {centered, Nx.squeeze(column_means, axes: [0]), all_mean}
  end

  defnp center_transform(kernel, kernel_fit_rows, kernel_fit_all) do
    predict_row_means = Nx.mean(kernel, axes: [1], keep_axes: true)
    kernel - kernel_fit_rows - predict_row_means + kernel_fit_all
  end

  # Eigenvectors of the centered kernel, sorted by decreasing eigenvalue and
  # sign-flipped so the largest absolute entry of each vector is positive.
  defnp top_eigen(kernel_centered, num_components) do
    {eigenvalues, eigenvectors} = Nx.LinAlg.eigh(kernel_centered, eps: 1.0e-8)

    order = Nx.argsort(eigenvalues, direction: :desc)
    eigenvalues = Nx.take(eigenvalues, order)[0..(num_components - 1)]
    eigenvectors = Nx.take(eigenvectors, order, axis: 1)[[.., 0..(num_components - 1)]]

    max_abs = eigenvectors |> Nx.abs() |> Nx.argmax(axis: 0, keep_axis: true)
    signs = eigenvectors |> Nx.take_along_axis(max_abs, axis: 0) |> Nx.sign()
    {eigenvalues, eigenvectors * signs}
  end

  defnp project(kernel_centered, eigenvalues, eigenvectors) do
    scaled_eigenvectors = eigenvectors / Nx.sqrt(eigenvalues)
    Nx.dot(kernel_centered, [1], scaled_eigenvectors, [0])
  end
end
