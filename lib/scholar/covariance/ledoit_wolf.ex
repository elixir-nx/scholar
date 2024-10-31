defmodule Scholar.Covariance.LedoitWolf do
  @moduledoc """
  Ledoit-Wolf is a particular form of shrinkage covariance estimator, where the shrinkage coefficient is computed using O.

  Ledoit and M. Wolf's formula as
  described in "A Well-Conditioned Estimator for Large-Dimensional
  Covariance Matrices", Ledoit and Wolf, Journal of Multivariate
  Analysis, Volume 88, Issue 2, February 2004, pages 365-411.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:covariance, :shrinkage, :location]}
  defstruct [:covariance, :shrinkage, :location]

  opts_schema = [
    block_size: [
      default: 1000,
      type: {:custom, Scholar.Options, :positive_number, []},
      doc: "Size of blocks into which the covariance matrix will be split."
    ],
    assume_centered: [
      default: false,
      type: :boolean,
      doc: """
        If `true`, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If `false`, data will be centered before computation.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)
  @doc """
  Estimate the shrunk Ledoit-Wolf covariance matrix.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:covariance` - Tensor of shape `{n_features, n_features}`. Estimated covariance matrix.

    * `:shrinkage` - Coefficient in the convex combination used for the computation of the shrunk estimate. Range is `[0, 1]`.

    * `:location` - Tensor of shape `{n_features,}`.
      Estimated location, i.e. the estimated mean.

  ## Examples

      iex> key = Nx.Random.key(0)
      iex> {x, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0.0, 0.0]), Nx.tensor([[0.4, 0.2], [0.2, 0.8]]), shape: {50}, type: :f32)
      iex> model = Scholar.Covariance.LedoitWolf.fit(x)
      iex> model.covariance
      #Nx.Tensor<
        f32[2][2]
        [
          [0.355768620967865, 0.17340737581253052],
          [0.17340737581253052, 1.0300586223602295]
        ]
      >
      iex> model.shrinkage
      #Nx.Tensor<
        f32
        0.15034136176109314
      >
      iex> model.location
      #Nx.Tensor<
        f32[2]
        [0.17184630036354065, 0.3276958167552948]
      >
      
      iex> key = Nx.Random.key(0)
      iex> {x, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0.0, 0.0, 0.0]), Nx.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.3, 1.0, 2.2]]), shape: {10}, type: :f32)
      iex> model = Scholar.Covariance.LedoitWolf.fit(x)
      iex> model.covariance
      #Nx.Tensor<
        f32[3][3]
        [
          [2.5945029258728027, 1.507835865020752, 1.1623677015304565],
          [1.507835865020752, 2.106797218322754, 1.181215524673462],
          [1.1623677015304565, 1.181215524673462, 1.460626482963562]
        ]
      >
      iex> model.shrinkage
      #Nx.Tensor<
        f32
        0.1908363550901413
      >
      iex> model.location 
      #Nx.Tensor<
        f32[3]
        [1.1228725910186768, 0.5419300198554993, 0.8678852319717407]
      >

      iex> key = Nx.Random.key(0)
      iex> {x, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0.0, 0.0, 0.0]), Nx.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.3, 1.0, 2.2]]), shape: {10}, type: :f32)
      iex> cov = Scholar.Covariance.LedoitWolf.fit(x, assume_centered: true)
      iex> cov.covariance
      #Nx.Tensor<
        f32[3][3]
        [
          [3.8574986457824707, 2.2048025131225586, 2.1504499912261963],
          [2.2048025131225586, 2.4572863578796387, 1.7215262651443481],
          [2.1504499912261963, 1.7215262651443481, 2.154898166656494]
        ]
      >
  """

  deftransform fit(x, opts \\ []) do
    fit_n(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(x, opts) do
    {x, location} = center(x, opts)

    {covariance, shrinkage} =
      ledoit_wolf(x, opts)

    %__MODULE__{
      covariance: covariance,
      shrinkage: shrinkage,
      location: location
    }
  end

  defnp center(x, opts) do
    x =
      case Nx.shape(x) do
        {_} -> Nx.new_axis(x, 1)
        _ -> x
      end

    location =
      if opts[:assume_centered] do
        Nx.broadcast(0, {Nx.axis_size(x, 1)})
      else
        Nx.mean(x, axes: [0])
      end

    {x - Nx.broadcast(location, x), location}
  end

  defnp ledoit_wolf(x, opts) do
    case Nx.shape(x) do
      {_n, 1} ->
        {
          Nx.pow(x, 2)
          |> Nx.mean()
          |> Nx.broadcast({1, 1}),
          0.0
        }

      _ ->
        ledoit_wolf_complex(x, opts)
    end
  end

  defnp ledoit_wolf_complex(x, opts) do
    n_features = Nx.axis_size(x, 1)
    shrinkage = ledoit_wolf_shrinkage(x, opts)
    emp_cov = empirical_covariance(x, opts)
    # git
    mu = Nx.sum(trace(emp_cov)) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    mask = Nx.iota(Nx.shape(shrunk_cov))
    selector = Nx.remainder(mask, n_features + 1) == 0
    shrunk_cov = Nx.select(selector, shrunk_cov + shrinkage * mu, shrunk_cov)
    {shrunk_cov, shrinkage}
  end

  defnp empirical_covariance(x, _opts) do
    n = Nx.axis_size(x, 0)

    covariance =
      Nx.transpose(x)
      |> Nx.dot(x)
      |> Nx.divide(n)

    case Nx.shape(covariance) do
      {} -> Nx.reshape(covariance, {1, 1})
      _ -> covariance
    end
  end

  defnp trace(x) do
    n = Nx.axis_size(x, 0)

    Nx.eye(n)
    |> Nx.multiply(x)
    |> Nx.sum()
  end

  defnp ledoit_wolf_shrinkage(x, opts) do
    case Nx.shape(x) do
      {_, 1} ->
        0

      {n} ->
        Nx.broadcast(x, {1, n})
        |> ledoit_wolf_shrinkage_complex(opts)

      _ ->
        ledoit_wolf_shrinkage_complex(x, opts)
    end
  end

  defnp ledoit_wolf_shrinkage_complex(x, opts) do
    {n_samples, n_features} = Nx.shape(x)
    {_, size} = Nx.type(x)
    block_size = opts[:block_size]
    n_splits = (n_features / block_size) |> Nx.floor() |> Nx.as_type({:s, size})

    x2 = Nx.pow(x, 2)
    emp_cov_trace = Nx.sum(x2, axes: [0]) / n_samples
    mu = Nx.sum(emp_cov_trace) / n_features
    beta = Nx.tensor(0.0, type: {:f, size})
    delta = Nx.tensor(0.0, type: {:f, size})
    i = Nx.tensor(0)
    block = Nx.iota({block_size})

    if n_splits > 0 do
      {beta, delta, _} =
        while {beta, delta, {x, x2, block, i, n_splits}}, i < n_splits do
          {block_size} = Nx.shape(block)
          j = Nx.tensor(0)

          {beta, delta, _} =
            while {beta, delta, {x, x2, block, i, j, n_splits}}, j < n_splits do
              {block_size} = Nx.shape(block)
              rows_from = block_size * i
              cols_from = block_size * j
              x2_t = Nx.transpose(x2)
              x_t = Nx.transpose(x)

              to_beta =
                Nx.slice_along_axis(x2_t, rows_from, block_size, axis: 0)
                |> Nx.dot(Nx.slice_along_axis(x2, cols_from, block_size, axis: 1))
                |> Nx.sum()

              beta = beta + to_beta

              to_delta =
                Nx.slice_along_axis(x_t, rows_from, block_size, axis: 0)
                |> Nx.dot(Nx.slice_along_axis(x, cols_from, block_size, axis: 1))
                |> Nx.pow(2)
                |> Nx.sum()

              delta = delta + to_delta

              {beta, delta, {x, x2, block, i, j + 1, n_splits}}
            end

          rows_from = block_size * i
          x2_t = Nx.transpose(x2)
          x_t = Nx.transpose(x)
          {m, n} = Nx.shape(x2)
          mask = Nx.iota({1, n}) |> Nx.tile([m]) |> Nx.reshape({m, n})
          mask = Nx.select(mask >= block_size * n_splits, 1, 0)

          to_beta =
            Nx.slice_along_axis(x2_t, rows_from, block_size, axis: 0)
            |> Nx.dot(Nx.multiply(x2, mask))
            |> Nx.sum()

          beta = beta + to_beta

          to_delta =
            Nx.slice_along_axis(x_t, rows_from, block_size, axis: 0)
            |> Nx.dot(Nx.multiply(x, mask))
            |> Nx.pow(2)
            |> Nx.sum()

          delta = delta + to_delta

          {beta, delta, {x, x2, block, i + 1, n_splits}}
        end

      j = Nx.tensor(0)

      {beta, delta, _} =
        while {beta, delta, {x, x2, block, j, n_splits}}, j < n_splits do
          {block_size} = Nx.shape(block)
          cols_from = block_size * j
          x2_t = Nx.transpose(x2)
          x_t = Nx.transpose(x)
          {rows, cols} = Nx.shape(x)

          mask =
            Nx.iota({1, cols}) |> Nx.tile([rows]) |> Nx.reshape({rows, cols}) |> Nx.transpose()

          mask = Nx.select(mask >= block_size * n_splits, 1, 0)

          to_beta =
            Nx.multiply(x2_t, mask)
            |> Nx.dot(Nx.slice_along_axis(x2, cols_from, block_size, axis: 1))
            |> Nx.sum()

          beta = beta + to_beta

          to_delta =
            Nx.multiply(x_t, mask)
            |> Nx.dot(Nx.slice_along_axis(x, cols_from, block_size, axis: 1))
            |> Nx.pow(2)
            |> Nx.sum()

          delta = delta + to_delta
          {beta, delta, {x, x2, block, j + 1, n_splits}}
        end

      {beta, delta}
    else
      {beta, delta}
    end

    x2_t = Nx.transpose(x2)
    x_t = Nx.transpose(x)
    {rows, cols} = Nx.shape(x)
    mask = Nx.iota({1, cols}) |> Nx.tile([rows]) |> Nx.reshape({rows, cols})
    mask = Nx.select(mask >= block_size * n_splits, 1, 0)
    mask_t = Nx.transpose(mask)

    to_delta =
      Nx.multiply(x_t, mask_t)
      |> Nx.dot(Nx.multiply(x, mask))
      |> Nx.pow(2)
      |> Nx.sum()

    delta = delta + to_delta
    delta = delta / Nx.pow(n_samples, 2)

    to_beta =
      Nx.multiply(x2_t, mask_t)
      |> Nx.dot(Nx.multiply(x2, mask))
      |> Nx.sum()

    beta = beta + to_beta
    beta = 1.0 / (n_features * n_samples) * (beta / n_samples - delta)
    delta = delta - 2.0 * mu * Nx.sum(emp_cov_trace) + n_features * Nx.pow(mu, 2)
    delta = delta / n_features
    beta = Nx.min(beta, delta)

    case beta do
      0 -> 0
      _ -> beta / delta
    end
  end
end
