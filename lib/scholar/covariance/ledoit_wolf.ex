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

      iex> x = Nx.tensor([[1, 2, 3], [6, 5, 4], [20, 0, 8]])
      iex> Scholar.Covariance.LedoitWolf.fit(x, block_size: 1)
      Nx.tensor(0)
  """

  deftransform fit(x, opts \\ []) do
    fit_n(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(x, opts) do
    location = Nx.mean(x, axes: [0])
    # {covariance, shrinkage} = 
      ledoit_wolf(x - Nx.broadcast(location, x), opts)

  end

  defnp ledoit_wolf(x, opts) do
    case Nx.shape(x) do
      {_n, 1}  -> Nx.pow(x, 2) 
        |> Nx.mean()
        |> Nx.broadcast({1,1})
      _       -> ledoit_wolf_complex(x, opts)
    end
  end

  defnp ledoit_wolf_complex(x, opts) do
    {_n_samples, _n_features} = Nx.shape(x)
    # shrinkage = 
    ledoit_wolf_shrinkage(x, opts)
    # emp_cov = empirical_covariance(x, opts)

  end

  # defnp empirical_covariance(x, _opts) do
  #   x = case Nx.shape(x) do
  #     {n} -> Nx.broadcast(x, {n, 1})
  #     _ -> x
  #   end
  # end

  defnp ledoit_wolf_shrinkage(x, opts) do
    case Nx.shape(x) do
      {_, 1} -> 0
      {n} -> Nx.broadcast(x, {n, 1}) 
        |> ledoit_wolf_shrinkage_complex(opts)
      _ -> ledoit_wolf_shrinkage_complex(x, opts)
    end
  end

  defnp ledoit_wolf_shrinkage_complex(x, opts) do
    {n_samples, n_features} = Nx.shape(x)
    {_, size} = Nx.type(x)
    n_splits = n_features / opts[:block_size] |> Nx.floor() |> Nx.as_type({:s, size})

    x2 = Nx.pow(x, 2)
    emp_cov_trace = Nx.sum(x2, axes: [0]) / n_samples
    mu = Nx.sum(emp_cov_trace) / n_features
    beta = 0.0
    delta = 0.0
    i = Nx.tensor(0)
    block_size = opts[:block_size]
    # {beta, delta, _} =
    #   while {beta, delta, {x, x2, block_size, i, n_splits}}, i < n_splits do
    #     j = Nx.tensor(0)
    #     {beta, delta, _} =
    #       while {beta, delta, {x, x2, block_size, i, j, n_splits}}, j < n_splits do
    #         rows_from = block_size * i
    #         cols_from = block_size * j
    #         x2_t = Nx.transpose(x2)
    #         x_t = Nx.transpose(x)
    #         beta = beta + Nx.dot(Nx.slice_along_axis(x2_t, rows_from, block_size, axis: 0),
    #                               Nx.slice_along_axis(x2, cols_from, block_size, axis: 1))
    #                       |> Nx.sum()
    #         delta = delta + Nx.dot(Nx.slice_along_axis(x_t, rows_from, block_size, axis: 0),
    #                               Nx.slice_along_axis(x, cols_from, block_size, axis: 1))
    #                       |> Nx.pow(2)
    #                       |> Nx.sum()

    #         {beta, delta, {x, x2, block_size, i, j + 1, n_splits}}
    #       end
    #       rows_from = block_size * i
    #       x2_t = Nx.transpose(x2)
    #       x_t = Nx.transpose(x)
    #       {_, n} = Nx.shape(x2)
    #       beta = beta + Nx.dot(Nx.slice_along_axis(x2_t, rows_from, block_size, axis: 0),
    #                             Nx.slice_along_axis(x2, block_size * n_splits, n, axis: 1))
    #                             |> Nx.sum()
    #       delta = delta + Nx.dot(Nx.slice_along_axis(x_t, rows_from, block_size, axis: 0),
    #                             Nx.slice_along_axis(x, block_size * n_splits, n, axis: 1))
    #                             |> Nx.pow(2)
    #                             |> Nx.sum()

    #     {beta, delta, {x, x2, block_size, i + 1, n_splits}}
    #   end
    j = Nx.tensor(0)
    {beta, delta, _} = 
      while {beta, delta, {x, x2, block_size, j, n_splits}}, j < n_splits do
        cols_from = block_size * j
        {_n, n_t} = Nx.shape(x2)
        x2_t = Nx.transpose(x2)
        x_t = Nx.transpose(x)
        {rows, cols} = Nx.shape(x_t)
        mask_base_t = Nx.iota({1, cols}) |> Nx.tile([rows]) |> Nx.reshape({rows, cols})
        {rows, cols} = Nx.shape(x)
        mask_base = Nx.iota({1, cols}) |> Nx.tile([rows]) |> Nx.reshape({rows, cols}) 
        row_mask_t = Nx.select(mask_base_t >= block_size * n_splits, 1, 0)
        col_mask = Nx.select(mask_base >= cols_from and mask_base < cols_from + block_size, 1, 0)
        beta = beta + Nx.dot(Nx.slice_along_axis(x2_t, block_size * n_splits, n_t, axis: 0),
                              Nx.slice_along_axis(x2, cols_from, block_size, axis: 1))
                              |> Nx.sum()
        # delta = delta + Nx.dot(Nx.slice_along_axis(x_t, block_size * n_splits, n_t, axis: 0),
        #                       Nx.slice_along_axis(x, cols_from, block_size, axis: 1))
        #                       |> Nx.pow(2)
        #                       |> Nx.sum()
        {beta, delta, {x, x2, block_size, j + 1, n_splits}}
      end
    x2_t = Nx.transpose(x2)
    x_t = Nx.transpose(x)
    delta = delta + Nx.dot(Nx.slice_along_axis(x_t, block_size * n_splits, n_features, axis: 0),
                          Nx.slice_along_axis(x, block_size * n_splits, n_features, axis: 1))
                    |> Nx.pow(2)
                    |> Nx.sum()
    delta = delta / Nx.pow(n_samples, 2)
    beta = beta + Nx.dot(Nx.slice_along_axis(x2_t, block_size * n_splits, n_samples, axis: 0),
                        Nx.slice_along_axis(x2, block_size * n_splits, n_features, axis: 1))
                    |> Nx.sum()
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
