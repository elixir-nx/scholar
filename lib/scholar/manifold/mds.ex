defmodule Scholar.Manifold.MDS do
  @moduledoc """
  TSNE (t-Distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction technique.

  ## References

  * [t-SNE: t-Distributed Stochastic Neighbor Embedding](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
  """
  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Metrics.Distance

  @derive {Nx.Container, containers: [:embedding, :stress, :n_iter]}
  defstruct [:embedding, :stress, :n_iter]

  opts_schema = [
    num_components: [
      type: :pos_integer,
      default: 2,
      doc: ~S"""
      Dimension of the embedded space.
      """
    ],
    metric: [
      type: :boolean,
      default: false,
      doc: ~S"""
      If `true`, use dissimilarities as metric distances in the embedding space.
      """
    ],
    normalized_stress: [
      type: :boolean,
      default: false,
      doc: ~S"""
      If `true`, normalize the stress by the sum of squared dissimilarities.
      """
    ],
    eps: [
      type: :float,
      default: 1.0e-3,
      doc: ~S"""
      Tolerance for stopping criterion.
      """
    ],
    max_iter: [
      type: :pos_integer,
      default: 300,
      doc: ~S"""
      Maximum number of iterations for the optimization.
      """
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Determines random number generation for centroid initialization.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ],
    n_init: [
      type: :pos_integer,
      default: 4,
      doc: ~S"""
      Number of times the embedding will be computed with different centroid seeds.
      The final embedding is the embedding with the lowest stress.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  # initialize x randomly or pass the init x earlier
  defnp smacof(dissimilarities, x, max_iter, opts) do
    # n = Nx.axis_size(dissimilarities, 0)
    similarities_flat = Nx.flatten(dissimilarities)
    similarities_flat_indices = lower_triangle_indices(dissimilarities)

    similarities_flat_w = Nx.take(similarities_flat, similarities_flat_indices)

    metric = if opts[:metric], do: 1, else: 0
    normalized_stress = if opts[:normalized_stress], do: 1, else: 0
    eps = opts[:eps]

    {{x, stress, i}, _} =
      while {{x, _stress = Nx.Constants.infinity(Nx.type(dissimilarities)), i = 0},
             {dissimilarities, max_iter, similarities_flat_indices, similarities_flat,
              similarities_flat_w, old_stress = Nx.Constants.infinity(Nx.type(dissimilarities)),
              metric, normalized_stress, eps, stop_value = 0}},
            i < max_iter and not stop_value do
        # i < 1 and not stop_value do
        dis = Distance.pairwise_euclidean(x)
        n = Nx.axis_size(dissimilarities, 0)

        disparities =
          if metric do
            dissimilarities
          else
            dis_flat = Nx.flatten(dis)

            dis_flat_indices = lower_triangle_indices(dis)

            n = Nx.axis_size(dis, 0)

            dis_flat_w = Nx.take(dis_flat, dis_flat_indices)
            # dis_flat_w = Nx.flatten(remove_main_diag(dis))

            disparities_flat_model =
              Scholar.Linear.IsotonicRegression.fit(similarities_flat_w, dis_flat_w)

            model = Scholar.Linear.IsotonicRegression.preprocess(disparities_flat_model)

            disparities_flat =
              Scholar.Linear.IsotonicRegression.predict(model, similarities_flat_w)

            # similarities_flat_indices

            # disparities = Nx.select(similarities_flat != 0, disparities_flat, disparities_flat)
            # {dis_flat, similarities_flat_indices, disparities_flat}

            disparities =
              Nx.indexed_put(
                dis_flat,
                Nx.new_axis(similarities_flat_indices, -1),
                disparities_flat
              )

            # disparities = Nx.reshape(dis, {n, n})

            disparities = Nx.reshape(disparities, {n, n})

            disparities * Nx.sqrt(n * (n - 1) / 2 / Nx.sum(disparities ** 2))
          end

        stress = Nx.sum((Nx.flatten(dis) - Nx.flatten(disparities)) ** 2) / 2

        stress =
          if normalized_stress do
            Nx.sqrt(stress / (Nx.sum(Nx.flatten(disparities) ** 2) / 2))
          else
            stress
          end

        dis = Nx.select(dis == 0, 1.0e-5, dis)
        ratio = disparities / dis
        b = -ratio
        b = Nx.put_diagonal(b, Nx.take_diagonal(b) + Nx.sum(ratio, axes: [1]))
        x = 1.0 / n * Nx.dot(b, x)

        dis = Nx.sum(Nx.sqrt(Nx.sum(x ** 2, axes: [1])))

        stop_value = if old_stress - stress / dis < eps, do: 1, else: 0

        old_stress = stress / dis

        {{x, stress, i + 1},
         {dissimilarities, max_iter, similarities_flat_indices, similarities_flat,
          similarities_flat_w, old_stress, metric, normalized_stress, eps, stop_value}}
      end

    {x, stress, i}
  end

  defnp mds_main_loop(dissimilarities, x, key, opts) do
    n_init = opts[:n_init]

    type = Nx.Type.merge(to_float_type(x), to_float_type(dissimilarities))
    dissimilarities = Nx.as_type(dissimilarities, type)
    x = Nx.as_type(x, type)

    {{best, best_stress, best_iter}, _} =
      while {{best = x, best_stress = Nx.Constants.infinity(type),
              best_iter = 0}, {n_init, dissimilarities, x, i = 0}},
            i < n_init do
        #           # i < 1 do
        {temp, stress, iter} = smacof(dissimilarities, x, opts[:max_iter], opts)
        # smacof(dissimilarities, x, opts[:max_iter], opts)

        {best, best_stress, best_iter} =
          if stress < best_stress, do: {temp, stress, iter}, else: {best, best_stress, best_iter}

        {best, best_stress, best_iter, {n_init, dissimilarities, x, i + 1}}
      end

    {best, best_stress, best_iter}
  end

  defnp mds_main_loop(dissimilarities, key, opts) do
    # key = opts[:key]
    n_init = opts[:n_init]
    max_iter = opts[:max_iter]
    num_samples = Nx.axis_size(dissimilarities, 0)

    type = to_float_type(dissimilarities)
    dissimilarities = Nx.as_type(dissimilarities, type)

    {dummy, new_key} =
      Nx.Random.uniform(key,
        shape: {num_samples, opts[:num_components]},
        type: type
      )

    dissimilarities = Distance.pairwise_euclidean(dissimilarities)

    {{best, best_stress, best_iter}, _} =
      while {{best = dummy, best_stress = Nx.Constants.infinity(type), best_iter = 0},
             {n_init, new_key, max_iter, dissimilarities, i = 0}},
            i < n_init do
        #           i < 1 do
        num_samples = Nx.axis_size(dissimilarities, 0)

        {x, new_key} =
          Nx.Random.uniform(new_key, shape: {num_samples, opts[:num_components]}, type: type)

        {temp, stress, iter} = smacof(dissimilarities, x, max_iter, opts)
        # smacof(dissimilarities, x, max_iter, opts)

        {best, best_stress, best_iter} =
          if stress < best_stress, do: {temp, stress, iter}, else: {best, best_stress, best_iter}

        {{best, best_stress, best_iter}, {n_init, new_key, max_iter, dissimilarities, i + 1}}
      end

    {best, best_stress, best_iter}
  end

  defn lower_triangle_indices(tensor) do
    n = Nx.axis_size(tensor, 0)

    temp = Nx.broadcast(Nx.s64(0), {div(n * (n - 1), 2)})

    {temp, _} =
      while {temp, {i = 0, j = 0}}, i < n ** 2 do
        {temp, j} =
          if Nx.remainder(i, n) < Nx.quotient(i, n) do
            temp = Nx.indexed_put(temp, Nx.new_axis(j, -1), i)
            {temp, j + 1}
          else
            {temp, j}
          end

        {temp, {i + 1, j}}
      end

    temp
  end

  @doc """
  Fits MDS for sample inputs `x`. It is simpyfied version of `fit/3` function.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  Returns struct with embedded data, stress value, and number of iterations for best run.

  ## Examples

      iex> x = Nx.iota({4,5})
      iex> Scholar.Manifold.MDS.fit(x)
      #Nx.Tensor<
        f32[4][2]
        [
          [-2197.154296875, 0.0],
          [-1055.148681640625, 0.0],
          [1055.148681640625, 0.0],
          [2197.154296875, 0.0]
        ]
      >
  """
  deftransform fit(x) do
    opts = NimbleOptions.validate!([], @opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_n(x, key, opts)
  end

  @doc """
  Fits MDS for sample inputs `x`. It is simpyfied version of `fit/3` function.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  Returns struct with embedded data, stress value, and number of iterations for best run.

  ## Examples

      iex> x = Nx.iota({4,5})
      iex> Scholar.Manifold.MDS.fit(x, num_components: 2)
      #Nx.Tensor<
        f32[4][2]
        [
          [-2197.154296875, 0.0],
          [-1055.148681640625, 0.0],
          [1055.148681640625, 0.0],
          [2197.154296875, 0.0]
        ]
      >
  """
  deftransform fit(x, opts) when is_list(opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_n(x, key, opts)
  end

  defnp fit_n(x, key, opts) do
    # mds_main_loop(x, key, opts)
    {best, best_stress, best_iter} = mds_main_loop(x, key, opts)
    %__MODULE__{embedding: best, stress: best_stress, n_iter: best_iter}
  end

  @doc """
  Fits MDS for sample inputs `x`. It is simpyfied version of `fit/3` function.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  Returns struct with embedded data, stress value, and number of iterations for best run.

  ## Examples

      iex> x = Nx.iota({4,5})
      iex> init = Nx.reverse(Nx.iota({4,5}))
      iex> Scholar.Manifold.MDS.fit(x, init)
      #Nx.Tensor<
        f32[4][2]
        [
          [-2197.154296875, 0.0],
          [-1055.148681640625, 0.0],
          [1055.148681640625, 0.0],
          [2197.154296875, 0.0]
        ]
      >
  """
  deftransform fit(x, init) do
    opts = NimbleOptions.validate!([], @opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_n(x, init, key, opts)
  end

  @doc """
  Fits MDS for sample inputs `x`. It is simpyfied version of `fit/3` function.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  Returns struct with embedded data, stress value, and number of iterations for best run.

  ## Examples

      iex> x = Nx.iota({4,5})
      iex> init = Nx.reverse(Nx.iota({4,5}))
      iex> Scholar.Manifold.MDS.fit(x, init, num_clusters: 3)
      #Nx.Tensor<
        f32[4][3]
        [
          [-2197.154296875, 0.0, 0.0],
          [-1055.148681640625, 0.0, 0.0],
          [1055.148681640625, 0.0, 0.0],
          [2197.154296875, 0.0, 0.0]
        ]
      >
  """
  deftransform fit(x, init, opts) when is_list(opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_n(x, init, key, opts)
  end

  defnp fit_n(x, init, key, opts) do
    {best, best_stress, best_iter} = mds_main_loop(x, init, key, opts)
    %__MODULE__{embedding: best, stress: best_stress, n_iter: best_iter}
  end

  # defn remove_main_diag_indices(tensor) do
  #   n = Nx.axis_size(tensor, 0)

  #   temp =
  #     Nx.broadcast(Nx.s64(0), {n})
  #     |> Nx.indexed_put(Nx.new_axis(0, -1), Nx.s64(1))
  #     |> Nx.tile([n - 1])

  #   Nx.iota({n * (n - 1)}) + Nx.cumulative_sum(temp)
  #   # indices = Nx.iota({n * (n - 1)}) + Nx.cumulative_sum(temp)
  #   # Nx.take(Nx.flatten(tensor), indices) |> Nx.reshape({n, n - 1})
  # end
end
