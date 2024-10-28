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
             :singular_values
           ]}
  defstruct [
    :components,
    :explained_variance,
    :explained_variance_ratio,
    :singular_values
  ]

  tsvd_schema = [
    num_components: [
      default: 2,
      type: :pos_integer,
      doc: "Desired dimensionality of output data."
    ],
    num_iter: [
      default: 5,
      type: :pos_integer,
      doc: "Number of iterations for randomized SVD solver."
    ],
    num_oversamples: [
      default: 10,
      type: :pos_integer,
      doc: "Number of oversamples for randomized SVD solver."
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Key for random tensor generation.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ]
  ]

  @tsvd_schema NimbleOptions.new!(tsvd_schema)

  @doc """
  Fit model on training data X.

  ## Options

  #{NimbleOptions.docs(@tsvd_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:components` - tensor of shape `{num_components, num_features}`
        The right singular vectors of the input data.

    * `:explained_variance` - tensor of shape `{num_components}`
        The variance of the training samples transformed by a projection to
        each component.

    * `:explained_variance_ratio` - tensor of shape `{num_components}`
        Percentage of variance explained by each of the selected components.

    * `:singular_values` -  ndarray of shape `{num_components}`
        The singular values corresponding to each of the selected components.

  ## Examples

      iex> key = Nx.Random.key(0)
      iex> x = Nx.tensor([[0, 0], [1, 0], [1, 1], [3, 3], [4, 4.5]])
      iex> tsvd = Scholar.Decomposition.TruncatedSVD.fit(x, num_components: 2, key: key)
      iex> tsvd.components
      #Nx.Tensor<
        f32[2][2]
        [
          [0.6871105432510376, 0.7265529036521912],
          [0.7265529036521912, -0.6871105432510376]
        ]
      >
      iex> tsvd.singular_values
      #Nx.Tensor<
        f32[2]
        [7.528080940246582, 0.7601959705352783]
      >

  """

  deftransform fit(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @tsvd_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_n(x, key, opts)
  end

  @doc """
  Fit model to X and perform dimensionality reduction on X.

  ## Options

  #{NimbleOptions.docs(@tsvd_schema)}

  ## Return Values

    X_new tensor of shape `{num_samples, num_components}` - reduced version of X. 

  ## Examples

      iex> key = Nx.Random.key(0)
      iex> x = Nx.tensor([[0, 0], [1, 0], [1, 1], [3, 3], [4, 4.5]])
      iex> Scholar.Decomposition.TruncatedSVD.fit_transform(x, num_components: 2, key: key)
      #Nx.Tensor<
        f32[5][2]
        [
          [0.0, 0.0],
          [0.6871105432510376, 0.7265529036521912],
          [1.413663387298584, 0.039442360401153564],
          [4.240990161895752, 0.1183270812034607],
          [6.017930030822754, -0.18578583002090454]
        ]
      >
      iex> key = Nx.Random.key(0)
      iex> x = Nx.tensor([[0, 0, 3], [1, 0, 3], [1, 1, 3], [3, 3, 3], [4, 4.5, 3]])
      iex> tsvd = Scholar.Decomposition.TruncatedSVD.fit_transform(x, num_components: 2, key: key)
      #Nx.Tensor<
        f32[5][2]
        [
          [1.9478826522827148, 2.260593891143799],
          [2.481153964996338, 1.906071662902832],
          [3.023407220840454, 1.352442979812622],
          [5.174456596374512, -0.46385863423347473],
          [6.521108150482178, -1.6488237380981445]
        ]
      >
  """

  deftransform fit_transform(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @tsvd_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_transform_n(x, key, opts)
  end

  defnp fit_n(x, key, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError, "expected x to have rank equal to: 2, got: #{inspect(Nx.rank(x))}"
    end

    num_components = opts[:num_components]
    {num_samples, num_features} = Nx.shape(x)

    cond do
      num_components > num_features ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_features = #{num_features}, got #{num_components}
              """

      num_components > num_samples ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_samples = #{num_samples}, got #{num_components}
              """

      true ->
        nil
    end

    {u, sigma, vt} = randomized_svd(x, key, opts)
    {_u, vt} = Scholar.Decomposition.Utils.flip_svd(u, vt)

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

  defnp fit_transform_n(x, key, opts) do
    module = fit_n(x, key, opts)
    Nx.dot(x, Nx.transpose(module.components))
  end

  defnp randomized_svd(m, key, opts) do
    num_components = opts[:num_components]
    num_oversamples = opts[:num_oversamples]
    num_iter = opts[:num_iter]
    n_random = num_components + num_oversamples
    {num_samples, num_features} = Nx.shape(m)

    transpose = num_samples < num_components

    m =
      if Nx.equal(transpose, 1) do
        Nx.transpose(m)
      else
        m
      end

    q = randomized_range_finder(m, key, size: n_random, num_iter: num_iter)

    q_t = Nx.transpose(q)
    b = Nx.dot(q_t, m)
    {uhat, s, vt} = Nx.LinAlg.svd(b)
    u = Nx.dot(q, uhat)
    vt = Nx.slice(vt, [0, 0], [num_components, num_features])
    s = Nx.slice(s, [0], [num_components])
    u = Nx.slice(u, [0, 0], [num_samples, num_components])

    if Nx.equal(transpose, 1) do
      {Nx.transpose(vt), s, Nx.transpose(u)}
    else
      {u, s, vt}
    end
  end

  defn randomized_range_finder(a, key, opts) do
    size = opts[:size]
    num_iter = opts[:num_iter]

    {_, a_cols} = Nx.shape(a)
    {q, _} = Nx.Random.normal(key, shape: {a_cols, size})
    a_t = Nx.transpose(a)

    {q, _} = Nx.LinAlg.qr(Nx.dot(a, q))
    {q, _} = Nx.LinAlg.qr(Nx.dot(a_t, q))

    {q, _} =
      while {q, {a, a_t, i = Nx.tensor(1), num_iter}}, Nx.less(i, num_iter) do
        {q, _} = Nx.LinAlg.qr(Nx.dot(a, q))
        {q, _} = Nx.LinAlg.qr(Nx.dot(a_t, q))
        {q, {a, a_t, i + 1, num_iter}}
      end

    {q, _} = Nx.LinAlg.qr(Nx.dot(a, q))
    q
  end
end
