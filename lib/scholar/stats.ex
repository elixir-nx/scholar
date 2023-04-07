defmodule Scholar.Stats do
  @moduledoc """
  Statistical functions

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  import Nx.Defn
  import Scholar.Shared

  general = [
    axes: [
      type: {:custom, Scholar.Options, :axes, []},
      default: [0],
      doc: """
      Axis to calculate the operation. If set to `nil` then
      the operation is performed on the whole tensor.
      """
    ],
    keep_axes: [
      type: :boolean,
      default: false,
      doc: "If set to true, the axes which are reduced are left."
    ]
  ]

  skew_schema =
    general ++
      [
        bias: [
          type: :boolean,
          default: true,
          doc: "If false, then the calculations are corrected for statistical bias."
        ]
      ]

  kurtosis_schema =
    general ++
      [
        bias: [
          type: :boolean,
          default: true,
          doc: "If false, then the calculations are corrected for statistical bias."
        ],
        variant: [
          type: {:in, [:fisher, :pearson]},
          default: :fisher,
          doc:
            "If :fisher then Fisher's definition is used, if :pearson then Pearson's definition is used."
        ]
      ]

  @moment_schema NimbleOptions.new!(general)
  @skew_schema NimbleOptions.new!(skew_schema)
  @kurtosis_schema NimbleOptions.new!(kurtosis_schema)

  @doc """
  Calculates the nth moment about the mean for a sample.

  ## Options

  #{NimbleOptions.docs(@moment_schema)}

  ## Examples

      iex> x = Nx.tensor([[3, 5, 3], [2, 6, 1], [9, 3, 2], [1, 6, 8]])
      iex> Scholar.Stats.moment(x, 2)
      #Nx.Tensor<
        f32[3]
        [9.6875, 1.5, 7.25]
      >
  """
  deftransform moment(tensor, moment, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @moment_schema)

    num_samples = num_samples(tensor, opts)

    moment_n(tensor, moment, num_samples, opts)
  end

  defnp moment_n(tensor, moment, num_samples, opts) do
    mean = Nx.mean(tensor, axes: opts[:axes], keep_axes: true) |> Nx.broadcast(tensor)
    Nx.sum((tensor - mean) ** moment, opts) / num_samples
  end

  @doc """
  Computes the sample skewness of a data set.

  ## Options

  #{NimbleOptions.docs(@skew_schema)}

  ## Examples

      iex> x = Nx.tensor([[3, 5, 3], [2, 6, 1], [9, 3, 2], [1, 6, 8]])
      iex> Scholar.Stats.skew(x)
      #Nx.Tensor<
        f32[3]
        [0.9794093370437622, -0.8164965510368347, 0.9220733642578125]
      >
  """

  deftransform skew(tensor, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @skew_schema)

    num_samples = num_samples(tensor, opts)

    skew_n(tensor, num_samples, opts)
  end

  defnp skew_n(tensor, num_samples, opts \\ []) do
    m2 = moment(tensor, 2, axes: opts[:axes], keep_axes: opts[:keep_axes])
    m3 = moment(tensor, 3, axes: opts[:axes], keep_axes: opts[:keep_axes])
    m2_mod = m2 ** (3 / 2)

    if opts[:bias] do
      m3 / m2_mod
    else
      m3 / m2_mod * Nx.sqrt(num_samples * (num_samples - 1)) / (num_samples - 2)
    end
  end

  @doc """
  Computes the kurtosis (Fisher or Pearson) of a dataset.

  ## Options

  #{NimbleOptions.docs(@kurtosis_schema)}

  ## Examples

      iex> x = Nx.tensor([[3, 5, 3], [2, 6, 1], [9, 3, 2], [1, 6, 8]])
      iex> Scholar.Stats.kurtosis(x)
      #Nx.Tensor<
        f32[3]
        [-0.7980852127075195, -1.0, -0.8394768238067627]
      >
  """
  deftransform kurtosis(tensor, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @kurtosis_schema)

    num_samples = num_samples(tensor, opts)

    kurtosis_n(tensor, num_samples, opts)
  end

  defnp kurtosis_n(tensor, num_samples, opts) do
    m2 = moment(tensor, 2, axes: opts[:axes], keep_axes: opts[:keep_axes])
    m4 = moment(tensor, 4, axes: opts[:axes], keep_axes: opts[:keep_axes])

    m2_mask = Nx.select(m2 == 0, Nx.Constants.nan(to_float_type(tensor)), m2)

    vals = m4 / m2_mask ** 2

    vals =
      cond do
        opts[:bias] or num_samples < 3 ->
          vals

        true ->
          1.0 / (num_samples - 2) / (num_samples - 3) *
            ((num_samples ** 2 - 1) * vals - 3 * (num_samples - 1) ** 2) + 3
      end

    case opts[:variant] do
      :fisher -> vals - 3
      :pearson -> vals
    end
  end

  deftransformp num_samples(tensor, opts) do
    Enum.product(Enum.map(opts[:axes] || Nx.axes(tensor), &Nx.axis_size(tensor, &1)))
  end
end
