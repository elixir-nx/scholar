defmodule Scholar.Covariance.LedoitWolf do
  @moduledoc """
  Ledoit-Wolf is a particular form of shrinkage covariance estimator, where the shrinkage coefficient is computed using O. Ledoit and M. Wolf’s formula.

  Ledoit and M. Wolf's formula as
  described in "A Well-Conditioned Estimator for Large-Dimensional
  Covariance Matrices", Ledoit and Wolf, Journal of Multivariate
  Analysis, Volume 88, Issue 2, February 2004, pages 365-411.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:covariance, :shrinkage, :location]}
  defstruct [:covariance, :shrinkage, :location]

  opts_schema = [
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

    * `:covariance` - Tensor of shape `{num_features, num_features}`. Estimated covariance matrix.

    * `:shrinkage` - Coefficient in the convex combination used for the computation of the shrunken estimate. Range is `[0, 1]`.

    * `:location` - Tensor of shape `{num_features,}`.
      Estimated location, i.e. the estimated mean.

  ## Examples

      iex> key = Nx.Random.key(0)
      iex> {x, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0.0, 0.0]), Nx.tensor([[0.4, 0.2], [0.2, 0.8]]), shape: {50}, type: :f32)
      iex> model = Scholar.Covariance.LedoitWolf.fit(x)
      iex> model.covariance
      #Nx.Tensor<
        f32[2][2]
        [
          [0.3557686507701874, 0.17340737581253052],
          [0.17340737581253052, 1.0300586223602295]
        ]
      >
      iex> model.shrinkage
      #Nx.Tensor<
        f32
        0.15034137666225433
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
          [2.5945029258728027, 1.5078359842300415, 1.1623677015304565],
          [1.5078359842300415, 2.106797456741333, 1.1812156438827515],
          [1.1623677015304565, 1.1812156438827515, 1.4606266021728516]
        ]
      >
      iex> model.shrinkage
      #Nx.Tensor<
        f32
        0.1908363401889801
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
    {x, location} = Scholar.Covariance.Utils.center(x, opts[:assume_centered])

    {covariance, shrinkage} =
      ledoit_wolf(x)

    %__MODULE__{
      covariance: covariance,
      shrinkage: shrinkage,
      location: location
    }
  end

  defnp ledoit_wolf(x) do
    case Nx.shape(x) do
      {_n, 1} ->
        {Nx.mean(x ** 2) |> Nx.reshape({1, 1}), 0.0}

      _ ->
        ledoit_wolf_shrinkage(x)
    end
  end

  defnp ledoit_wolf_shrinkage(x) do
    case Nx.shape(x) do
      {_, 1} ->
        0

      {n} ->
        Nx.reshape(x, {1, n})
        |> ledoit_wolf_shrinkage_complex()

      _ ->
        ledoit_wolf_shrinkage_complex(x)
    end
  end

  defnp ledoit_wolf_shrinkage_complex(x) do
    {num_samples, num_features} = Nx.shape(x)
    emp_cov = Scholar.Covariance.Utils.empirical_covariance(x)

    emp_cov_trace = Scholar.Covariance.Utils.trace(emp_cov)
    mu = Nx.sum(emp_cov_trace) / num_features

    flatten_delta = Nx.flatten(emp_cov)

    indices =
      Nx.shape(flatten_delta)
      |> Nx.iota()

    subtract = Nx.select(Nx.remainder(indices, num_features + 1) == 0, mu, 0)

    delta =
      (flatten_delta - subtract)
      |> Nx.pow(2)
      |> Nx.sum()

    delta = delta / num_features

    x2 = Nx.pow(x, 2)

    beta =
      (Nx.dot(x2, [0], x2, [0]) / num_samples - emp_cov ** 2)
      |> Nx.sum()
      |> Nx.divide(num_features * num_samples)

    beta = Nx.min(beta, delta)
    shrinkage = beta / delta

    shrunk_cov = (1.0 - shrinkage) * emp_cov
    mask = Nx.iota(Nx.shape(shrunk_cov))
    selector = Nx.remainder(mask, num_features + 1) == 0
    shrunk_cov = Nx.select(selector, shrunk_cov + shrinkage * mu, shrunk_cov)

    {shrunk_cov, shrinkage}
  end
end
