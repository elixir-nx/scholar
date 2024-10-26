defmodule Scholar.Covariance.LedoitWolf do
  import Nx.Defn

  opts = [
    block_size: [
      default: 1000,
      type: {:custom, Scholar.Options, :positive_number, []}
    ],
  ]

  @opts_schema NimbleOptions.new!(opts)
  @doc """
  ## Examples

      iex> key = Nx.Random.key(0)
      iex> {x, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0.0, 0.0]), Nx.tensor([[0.4, 0.2], [0.2, 0.8]]), shape: {50}, type: :f32)
      iex> Scholar.Covariance.LedoitWolf.fit(x)
      {#Nx.Tensor<
        f32[2][2]
        [
          [0.355768620967865, 0.17340737581253052],
          [0.17340737581253052, 1.0300586223602295]
        ]
      >,
      #Nx.Tensor<
        f32
        0.15034136176109314
      >}
      iex> {x, _new_key} = Nx.Random.normal(key, 1, 6, shape: {10, 2}, type: :f32)
      iex> Scholar.Covariance.LedoitWolf.fit(x)
      iex> 
  """

  deftransform fit(x, opts \\ []) do
    fit_n(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(x, opts) do
    location = Nx.mean(x, axes: [0])
    {covariance, shrinkage} = 
      ledoit_wolf(x - Nx.broadcast(location, x), opts)
  end

  defnp ledoit_wolf(x, opts) do
    case Nx.shape(x) do
      {_n, 1}  -> {
        Nx.pow(x, 2) 
        |> Nx.mean()
        |> Nx.broadcast({1,1}), 
        0.0
      }
      _       -> ledoit_wolf_complex(x, opts)
    end
  end

  defnp ledoit_wolf_complex(x, opts) do
    n_features = Nx.axis_size(x, 1)
    shrinkage = ledoit_wolf_shrinkage(x, opts)
    emp_cov = empirical_covariance(x, opts)
    mu = Nx.sum(trace(emp_cov)) / n_features#git
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    mask = Nx.iota(Nx.shape(shrunk_cov))
    selector = Nx.remainder(mask, n_features + 1) == 0
    shrunk_cov = Nx.select(selector, shrunk_cov + shrinkage * mu, shrunk_cov)
    {shrunk_cov, shrinkage}
  end

  defnp empirical_covariance(x, _opts) do
    x = case Nx.shape(x) do
      {n} -> Nx.broadcast(x, {n, 1})
      _ -> x
    end
    n = Nx.axis_size(x, 0)
    covariance = Nx.transpose(x) 
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
      {_, 1} -> 0
      {n} -> Nx.broadcast(x, {1, n}) 
        |> ledoit_wolf_shrinkage_complex(opts)
      _ -> ledoit_wolf_shrinkage_complex(x, opts)
    end
  end

  defnp ledoit_wolf_shrinkage_complex(x, opts) do
    {n_samples, n_features} = Nx.shape(x)
    {_, size} = Nx.type(x)
    block_size = opts[:block_size]
    n_splits = n_features / block_size |> Nx.floor() |> Nx.as_type({:s, size})

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
              to_beta = Nx.slice_along_axis(x2_t, rows_from, block_size, axis: 0)
                        |> Nx.dot(Nx.slice_along_axis(x2, cols_from, block_size, axis: 1))
                        |> Nx.sum()
              beta = beta + to_beta
              to_delta = Nx.slice_along_axis(x_t, rows_from, block_size, axis: 0)
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

            to_beta = Nx.slice_along_axis(x2_t, rows_from, block_size, axis: 0)
                        |> Nx.dot(Nx.multiply(x2, mask))
                        |> Nx.sum()
            beta = beta + to_beta
            to_delta = Nx.slice_along_axis(x_t, rows_from, block_size, axis: 0)
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
          mask = Nx.iota({1, cols}) |> Nx.tile([rows]) |> Nx.reshape({rows, cols}) |> Nx.transpose()
          mask = Nx.select(mask >= block_size * n_splits, 1, 0)
          to_beta = Nx.multiply(x2_t, mask)
                    |> Nx.dot(Nx.slice_along_axis(x2, cols_from, block_size, axis: 1))
                    |> Nx.sum()
          beta = beta + to_beta
          to_delta = Nx.multiply(x_t, mask)
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
    to_delta = Nx.multiply(x_t, mask_t)
              |> Nx.dot(Nx.multiply(x, mask))
              |> Nx.pow(2)
              |> Nx.sum()
    delta = delta + to_delta
    delta = delta / Nx.pow(n_samples, 2)
    to_beta = Nx.multiply(x2_t, mask_t)
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
