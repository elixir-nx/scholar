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
      default: true,
      doc: ~S"""
      If `true`, use dissimilarities as metric distances in the embedding space.
      """
    ],
    normalized_stress: [
      type: :boolean,
      default: false,
      doc: ~S"""
      If `true`, normalize the stress by the sum of squared dissimilarities.
      Only valid if `metric` is `false`.
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
    similarities_flat = Nx.flatten(dissimilarities)
    similarities_flat_indices = lower_triangle_indices(dissimilarities)

    similarities_flat_w = Nx.take(similarities_flat, similarities_flat_indices)

    metric = if opts[:metric], do: 1, else: 0
    normalized_stress = if opts[:normalized_stress], do: 1, else: 0
    eps = opts[:eps]
    n = Nx.axis_size(dissimilarities, 0)

    {{x, stress, i}, _} =
      while {{x, _stress = Nx.Constants.infinity(Nx.type(dissimilarities)), i = 0},
             {dissimilarities, max_iter, similarities_flat_indices, similarities_flat,
              similarities_flat_w, old_stress = Nx.Constants.infinity(Nx.type(dissimilarities)),
              metric, normalized_stress, eps, stop_value = 0}},
            i < max_iter and not stop_value do
        dis = Distance.pairwise_euclidean(x)

        disparities =
          if metric do
            dissimilarities
          else
            dis_flat = Nx.flatten(dis)

            dis_flat_indices = lower_triangle_indices(dis)

            dis_flat_w = Nx.take(dis_flat, dis_flat_indices)

            disparities_flat_model =
              Scholar.Linear.IsotonicRegression.fit(similarities_flat_w, dis_flat_w,
                increasing: true
              )

            model = Scholar.Linear.IsotonicRegression.special_preprocess(disparities_flat_model)

            disparities_flat =
              Scholar.Linear.IsotonicRegression.predict(model, similarities_flat_w)

            disparities =
              Nx.indexed_put(
                dis_flat,
                Nx.new_axis(similarities_flat_indices, -1),
                disparities_flat
              )

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
        x = Nx.dot(b, x) * (1.0 / n)

        dis = Nx.sum(Nx.sqrt(Nx.sum(x ** 2, axes: [1])))

        stop_value = if old_stress - stress / dis < eps, do: 1, else: 0

        old_stress = stress / dis

        {{x, stress, i + 1},
         {dissimilarities, max_iter, similarities_flat_indices, similarities_flat,
          similarities_flat_w, old_stress, metric, normalized_stress, eps, stop_value}}
      end

    {x, stress, i}
  end

  defnp mds_main_loop(dissimilarities, x, _key, opts) do
    n_init = opts[:n_init]

    type = Nx.Type.merge(to_float_type(x), to_float_type(dissimilarities))
    dissimilarities = Nx.as_type(dissimilarities, type)
    x = Nx.as_type(x, type)

    dissimilarities = Distance.pairwise_euclidean(dissimilarities)

    {{best, best_stress, best_iter}, _} =
      while {{best = x, best_stress = Nx.Constants.infinity(type), best_iter = 0},
             {n_init, dissimilarities, x, i = 0}},
            i < n_init do
        {temp, stress, iter} = smacof(dissimilarities, x, opts[:max_iter], opts)

        {best, best_stress, best_iter} =
          if stress < best_stress, do: {temp, stress, iter}, else: {best, best_stress, best_iter}

        {{best, best_stress, best_iter}, {n_init, dissimilarities, x, i + 1}}
      end

    {best, best_stress, best_iter}
  end

  defnp mds_main_loop(dissimilarities, key, opts) do
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
        num_samples = Nx.axis_size(dissimilarities, 0)

        {x, new_key} =
          Nx.Random.uniform(new_key, shape: {num_samples, opts[:num_components]}, type: type)

        {temp, stress, iter} = smacof(dissimilarities, x, max_iter, opts)

        {best, best_stress, best_iter} =
          if stress < best_stress, do: {temp, stress, iter}, else: {best, best_stress, best_iter}

        {{best, best_stress, best_iter}, {n_init, new_key, max_iter, dissimilarities, i + 1}}
      end

    {best, best_stress, best_iter}
  end

  defnp lower_triangle_indices(tensor) do
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
      iex> key = Nx.Random.key(42)
      iex> Scholar.Manifold.MDS.fit(x, key: key)
      %Scholar.Manifold.MDS{
        embedding: Nx.tensor(
          [
            [16.3013916015625, -3.444634437561035],
            [5.866805553436279, 1.6378790140151978],
            [-5.487184524536133, 0.5837264657020569],
            [-16.681013107299805, 1.2230290174484253]
          ]
        ),
        stress: Nx.tensor(
          0.3993147909641266
        ),
        n_iter: Nx.tensor(
          23
        )
      }
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
      iex> key = Nx.Random.key(42)
      iex> Scholar.Manifold.MDS.fit(x, num_components: 2, key: key)
      %Scholar.Manifold.MDS{
        embedding: Nx.tensor(
          [
            [16.3013916015625, -3.444634437561035],
            [5.866805553436279, 1.6378790140151978],
            [-5.487184524536133, 0.5837264657020569],
            [-16.681013107299805, 1.2230290174484253]
          ]
        ),
        stress: Nx.tensor(
          0.3993147909641266
        ),
        n_iter: Nx.tensor(
          23
        )
      }
  """
  deftransform fit(x, opts) when is_list(opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    fit_n(x, key, opts)
  end

  defnp fit_n(x, key, opts) do
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
      iex> key = Nx.Random.key(42)
      iex> init = Nx.reverse(Nx.iota({4,2}))
      iex> Scholar.Manifold.MDS.fit(x, init)
      %Scholar.Manifold.MDS{
        embedding: Nx.tensor(
          [
            [11.858541488647461, 11.858541488647461],
            [3.9528470039367676, 3.9528470039367676],
            [-3.9528470039367676, -3.9528470039367676],
            [-11.858541488647461, -11.858541488647461]
          ]
        ),
        stress: Nx.tensor(
          0.0
        ),
        n_iter: Nx.tensor(
          3
        )
      }
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
      iex> key = Nx.Random.key(42)
      iex> init = Nx.reverse(Nx.iota({4,3}))
      iex> Scholar.Manifold.MDS.fit(x, init, num_components: 3, key: key)
      %Scholar.Manifold.MDS{
        embedding: Nx.tensor(
          [
            [9.682458877563477, 9.682458877563477, 9.682458877563477],
            [3.2274858951568604, 3.2274858951568604, 3.2274858951568604],
            [-3.2274863719940186, -3.2274863719940186, -3.2274863719940186],
            [-9.682458877563477, -9.682458877563477, -9.682458877563477]
          ]
        ),
        stress: Nx.tensor(
          9.094947017729282e-12
        ),
        n_iter: Nx.tensor(
          3
        )
      }
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
end
