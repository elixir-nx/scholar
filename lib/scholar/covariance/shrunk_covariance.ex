defmodule Scholar.Covariance.ShrunkCovariance do
  @moduledoc """
  Covariance estimator with shrinkage.


  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:covariance, :location]}
  defstruct [:covariance, :location]

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
    ],
    shrinkage: [
      default: 0.1,
      type: :float,
      doc: "Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1]."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)
  @doc """
  Fit the shrunk covariance model to `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:
    * `:covariance` - Tensor of shape `{num_features, num_features}`. Estimated covariance matrix.
    * `:location` - Tensor of shape `{num_features,}`.
      Estimated location, i.e. the estimated mean.

  ## Examples

      iex> key = Nx.Random.key(0)
      iex> {x, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0.0, 0.0]), Nx.tensor([[0.8, 0.3], [0.2, 0.4]]), shape: {10}, type: :f32)
      iex> model = Scholar.Covariance.ShrunkCovariance.fit(x)
      iex> model.covariance
      #Nx.Tensor<
        f32[2][2]
        [
          [0.7721845507621765, 0.19141492247581482],
          [0.19141492247581482, 0.33952537178993225]
        ]
      >
      iex> model.location
      #Nx.Tensor<
        f32[2]
        [0.18202415108680725, -0.09216632694005966]
      >
      iex> key = Nx.Random.key(0)
      iex> {x, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0.0, 0.0]), Nx.tensor([[0.8, 0.3], [0.2, 0.4]]), shape: {10}, type: :f32)
      iex> model = Scholar.Covariance.ShrunkCovariance.fit(x, shrinkage: 0.4)
      iex> model.covariance
      #Nx.Tensor<
        f32[2][2]
        [
          [0.7000747323036194, 0.1276099532842636],
          [0.1276099532842636, 0.41163527965545654]
        ]
      >
      iex> model.location
      #Nx.Tensor<
        f32[2]
        [0.18202415108680725, -0.09216632694005966]
      >


  """

  deftransform fit(x, opts \\ []) do
    fit_n(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(x, opts) do
    shrinkage = opts[:shrinkage]

    if shrinkage < 0 or shrinkage > 1 do
      raise ArgumentError,
            """
            expected :shrinkage option to be in [0, 1] range, \
            got shrinkage: #{inspect(Nx.shape(x))}\
            """
    end

    {x, location} = Scholar.Covariance.Utils.center(x, opts[:assume_centered])

    covariance =
      Scholar.Covariance.Utils.empirical_covariance(x)
      |> shrunk_covariance(shrinkage)

    %__MODULE__{
      covariance: covariance,
      location: location
    }
  end

  defnp shrunk_covariance(emp_cov, shrinkage) do
    num_features = Nx.axis_size(emp_cov, 1)
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    emp_cov_trace = Scholar.Covariance.Utils.trace(emp_cov)
    mu = Nx.sum(emp_cov_trace) / num_features

    mask = Nx.iota(Nx.shape(shrunk_cov))
    selector = Nx.remainder(mask, num_features + 1) == 0

    Nx.select(selector, shrunk_cov + shrinkage * mu, shrunk_cov)
  end
end
