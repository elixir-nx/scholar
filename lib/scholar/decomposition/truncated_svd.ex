defmodule Scholar.Decomposition.TruncatedSVD do
  @moduledoc """
  Dimensionality reduction using truncated SVD (aka LSA).

  This transformer performs linear dimensionality reduction by means of
  truncated singular value decomposition (SVD).
  """
  import Nx.Defn

  @derive {Nx.Container,
           containers: [
             :components,
             :explained_variance,
             :explained_variance_ratio,
             :singular_values,
           ]}
  defstruct [
    :components,
    :explained_variance,
    :explained_variance_ratio,
    :singular_values,
  ]

  tsvd_schema = [
    n_components: [
      default: 2,
      type: :pos_integer,
      doc: "Number of components to keep"
    ],
    n_oversamples: [
      default: 10,
      type: :pos_integer,
      doc: "Number of additional random vectors to sample"
    ],
    transpose: [
      default: :auto,
      type: {:in, [:true, :false, :auto]},
      doc: "Whether to transpose the input matrix"
    ],
    seed: [
      default: 0,
      type: :integer,
      doc: "Seed for random number generation"
    ], 
    n_iter: [
      type: :pos_integer,
    ],
  ]

  @tsvd_schema NimbleOptions.new!(tsvd_schema)

  @doc """
  Fit model on training data X.

  ## Options

  #{NimbleOptions.docs(@tsvd_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:components` - tensor of shape `{n_components, n_features}`
        The right singular vectors of the input data.

    * `:explained_variance` - tensor of shape (n_components,)
        The variance of the training samples transformed by a projection to
        each component.

    * `:explained_variance_ratio` - 

    * `:singular_values` - 

  ## Examples

  """

  deftransform fit(x, opts \\ []) do
    fit_n(x, NimbleOptions.validate!(opts, @tsvd_schema))
  end

  deftransform fit_transform(x, opts \\ []) do
    fit_transform_n(x, NimbleOptions.validate!(opts, @tsvd_schema))
  end

  defnp fit_n(x, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError, "expected x to have rank equal to: 2, got: #{inspect(Nx.rank(x))}"
    end
    transpose = opts[:transpose]

    {u, sigma, vt} = randomized_svd(x, opts)
    {n_samples, _} = Nx.shape(x)

    transpose =
      case transpose do
        :auto -> n_samples < opts[:n_components]
        _ -> transpose
      end

    {_u, vt} =
      svd_flip(u, vt, not transpose)
    x_transformed = Nx.dot(x, Nx.transpose(vt))
    explained_variance = Nx.variance(x_transformed, axes: [0])
    full_variance = Nx.variance(x, axes: [0]) |> Nx.sum()
    explained_variance_ratio = explained_variance / full_variance

    %__MODULE__{
      components: vt,
      explained_variance: explained_variance,
      explained_variance_ratio: explained_variance_ratio,
      singular_values: sigma
    }
  end

  defnp fit_transform_n(x, opts) do
    module = fit_n(x, opts)
    Nx.dot(x, Nx.transpose(module.components))
  end

  defnp randomized_svd(m, opts) do
    n_components = opts[:n_components]
    n_oversamples = opts[:n_oversamples]
    transpose = opts[:transpose]
    n_random = n_components + n_oversamples
    {n_samples, n_features} = Nx.shape(m)
    n_iter = 
      case opts[:n_iter] do 
        :nil -> Nx.select(n_components < 0.1 * Nx.min(n_samples, n_features), 7, 4)
        any -> any
      end

    transpose =
      case transpose do
        :auto -> n_samples < n_components
        _ -> transpose
      end

    m =
      case transpose do
        true -> Nx.transpose(m)
        false -> m
      end

    q = randomized_range_finder(m, size: n_random, n_iter: n_iter, seed: opts[:seed])

    q_t = Nx.transpose(q)
    b = Nx.dot(q_t, m)
    {uhat, s, vt} = Nx.LinAlg.svd(b)
    u = Nx.dot(q, uhat)
    vt = Nx.slice(vt, [0, 0], [n_components, n_features])
    s = Nx.slice(s, [0], [n_components])
    u = Nx.slice(u, [0, 0], [n_samples, n_components])

    case transpose do
      true -> {Nx.transpose(vt), s, Nx.transpose(u)}
      false -> {u, s, vt}
    end
  end

  defn randomized_range_finder(a, opts) do
    size = opts[:size]
    n_iter = opts[:n_iter]
    seed = opts[:seed]

    key = Nx.Random.key(seed) 

    {_, a_cols} = Nx.shape(a)
    {q, _} = Nx.Random.normal(key, shape: {a_cols, size})
    a_t = Nx.transpose(a)

    {q, _} = Nx.LinAlg.qr(Nx.dot(a, q))
    {q, _} = Nx.LinAlg.qr(Nx.dot(a_t, q))

    {q, _a, _a_t, _i, _n_iter} =
      while {q, a, a_t, i = Nx.tensor(1), n_iter}, Nx.less(i, n_iter) do
        {q, _} = Nx.LinAlg.qr(Nx.dot(a, q))
        {q, _} = Nx.LinAlg.qr(Nx.dot(a_t, q))
        {q, a, a_t, i + 1, n_iter}
      end

    {q, _} = Nx.LinAlg.qr(Nx.dot(a, q))
    q
  end

  defnp svd_flip(u, v, u_based_decision) do
    base =
      if Nx.equal(u_based_decision, 1) do
        Nx.transpose(u)
      else
        v
      end

    max_abs_cols = base 
      |> Nx.abs()
      |> Nx.argmax(axis: 1)

    {rows, cols} = Nx.shape(base)
    shift = Nx.iota({rows})
    indices = max_abs_cols + shift * cols

    signs = base
      |> Nx.flatten()
      |> Nx.take(indices)
      |> Nx.sign()

    if Nx.equal(u_based_decision, 1) do
      {u * Nx.new_axis(signs, 0), v * Nx.new_axis(signs, 1)}
    else
      {u * Nx.new_axis(signs, 1), v * Nx.new_axis(signs, 0)}
    end
  end
end

# Test the updated TruncatedSVD with seed
# x = Nx.tensor([[0, 0], [1, 0], [1, 1], [3, 3], [4, 4.5]], type: {:f, 64})
# Scholar.Decomposition.TruncatedSVD.fit(x, transpose: false, n_components: 1, n_iter: 7)
#OUTPUT: 
# {#Nx.Tensor<
#    f64[5][1]
#    [
#      [0.0],
#      [0.6871101145887069],
#      [1.4136634095574203],
#      [4.240990228672261],
#      [6.017930285714038]
#    ]
#  >,
#  #Nx.Tensor<
#    f64[1]
#    [5.223928048654531]
#  >,
#  #Nx.Tensor<
#    f64[1]
#    [0.9819413625290471]
#  >,
#  #Nx.Tensor<
#    f64[1]
#    [7.5280803030880605]
#  >}