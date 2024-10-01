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
    n_components: [
      default: 2,
      type: :pos_integer,
      doc: "Desired dimensionality of output data."
    ],
    n_iter: [
      default: 5,
      type: :pos_integer,
      doc: "Number of iterations for randomized SVD solver."
    ],
    n_oversamples: [
      default: 10,
      type: :pos_integer,
      doc: "Number of oversamples for randomized SVD solver."
    ],
    seed: [
      default: 0,
      type: :integer,
      doc: "Seed for random tensor generation."
    ]
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

    * `:explained_variance` - tensor of shape `{n_components}`
        The variance of the training samples transformed by a projection to
        each component.

    * `:explained_variance_ratio` - tensor of shape `{n_components}`
        Percentage of variance explained by each of the selected components.

    * `:singular_values` -  ndarray of shape `{n_components}`
        The singular values corresponding to each of the selected components.

  ## Examples

      iex> x = Nx.tensor([[0, 0], [1, 0], [1, 1], [3, 3], [4, 4.5]])
      iex> tsvd = Scholar.Decomposition.TruncatedSVD.fit(x, n_components: 2)
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
    fit_n(x, NimbleOptions.validate!(opts, @tsvd_schema))
  end

  @doc """
  Fit model to X and perform dimensionality reduction on X.

  ## Options

  #{NimbleOptions.docs(@tsvd_schema)}

  ## Return Values

    X_new tensor of shape `{n_samples, n_components}` - reduced version of X. 

  ## Examples

      iex> x = Nx.tensor([[0, 0], [1, 0], [1, 1], [3, 3], [4, 4.5]])
      iex> Scholar.Decomposition.TruncatedSVD.fit_transform(x, n_components: 2)
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
  """

  deftransform fit_transform(x, opts \\ []) do
    fit_transform_n(x, NimbleOptions.validate!(opts, @tsvd_schema))
  end

  defnp fit_n(x, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError, "expected x to have rank equal to: 2, got: #{inspect(Nx.rank(x))}"
    end

    n_components = opts[:n_components]
    {_n_samples, n_features} = Nx.shape(x)

    cond do
      n_components > n_features ->
        raise ArgumentError,
              """
              n_components must be less than or equal to \
              n_features = #{n_features}, got #{n_components}
              """

      true ->
        nil
    end

    {u, sigma, vt} = randomized_svd(x, opts)
    {_u, vt} = Scholar.Decomposition.PCA.flip_svd(u, vt)

    x_transformed = Nx.dot(x, [1], vt, [0])
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
    Nx.dot(x, [1], module.components, [0])
  end

  defnp randomized_svd(m, opts) do
    n_components = opts[:n_components]
    n_oversamples = opts[:n_oversamples]
    n_iter = opts[:n_iter]
    n_random = n_components + n_oversamples
    {n_samples, n_features} = Nx.shape(m)

    transpose = n_samples < n_components

    m =
      if Nx.equal(transpose, 1) do
        Nx.transpose(m)
      else
        m
      end

    q = randomized_range_finder(m, size: n_random, n_iter: n_iter, seed: opts[:seed])

    q_t = Nx.transpose(q)
    b = Nx.dot(q_t, m)
    {uhat, s, vt} = Nx.LinAlg.svd(b)
    u = Nx.dot(q, uhat)
    vt = Nx.slice(vt, [0, 0], [n_components, n_features])
    s = Nx.slice(s, [0], [n_components])
    u = Nx.slice(u, [0, 0], [n_samples, n_components])

    if Nx.equal(transpose, 1) do
      {Nx.transpose(vt), s, Nx.transpose(u)}
    else
      {u, s, vt}
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

    {q, _} =
      while {q, {a, a_t, i = Nx.tensor(1), n_iter}}, Nx.less(i, n_iter) do
        {q, _} = Nx.LinAlg.qr(Nx.dot(a, q))
        {q, _} = Nx.LinAlg.qr(Nx.dot(a_t, q))
        {q, {a, a_t, i + 1, n_iter}}
      end

    {q, _} = Nx.LinAlg.qr(Nx.dot(a, q))
    q
  end
end
