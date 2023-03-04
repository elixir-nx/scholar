defmodule Scholar.Stats do
  @moduledoc """
  Statistical functions

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  import Nx.Defn

  general = [
    axis: [
      type: {:or, [:non_neg_integer, :atom]},
      default: 0,
      doc: """
      Axis to calculate the operation. If set to `nil` then
      the operation is performed on the whole tensor.
      """
    ],
    keep_axis: [
      type: :boolean,
      default: false,
      doc: "If set to true, the axis which is reduced is left."
    ]
  ]

  moment_schema =
    general ++
      [
        moment: [
          type: :pos_integer,
          default: 1,
          doc: "Order of central moment that is returned."
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

  @moment_schema NimbleOptions.new!(moment_schema)
  @skew_schema NimbleOptions.new!(skew_schema)
  @kurtosis_schema NimbleOptions.new!(kurtosis_schema)

  @doc """
  Calculate the nth moment about the mean for a sample.

  ## Options

  #{NimbleOptions.docs(@moment_schema)}

  ## Return Values

    The appropriate moment along the given axis or whole tensor.

  ## Examples

      iex> x = Nx.tensor([[3, 5, 3], [2, 6, 1], [9, 3, 2], [1, 6, 8]])
      iex> Scholar.Stats.moment(x, moment: 2)
      #Nx.Tensor<
        f32[3]
        [9.6875, 1.5, 7.25]
      >
  """
  deftransform moment(tensor, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @moment_schema)

    axis =
      if axis_opt = opts[:axis] do
        Nx.Shape.normalize_axis(Nx.shape(tensor), axis_opt, Nx.names(tensor))
      end

    opts = Keyword.put(opts, :axis, if(axis == nil, do: nil, else: [axis]))
    num_samples = if opts[:axis] == nil, do: Nx.size(tensor), else: Nx.axis_size(tensor, 0)
    moment_n(tensor, num_samples, opts)
  end

  defnp moment_n(tensor, num_samples, opts) do
    mean = Nx.mean(tensor, axes: opts[:axis], keep_axes: opts[:keep_axis])

    Nx.sum((tensor - mean) ** opts[:moment], axes: opts[:axis], keep_axes: opts[:keep_axis]) /
      num_samples
  end

  @doc """
  Compute the sample skewness of a data set.

  ## Options

  #{NimbleOptions.docs(@skew_schema)}

  ## Return Values

    The skewness of values along an axis, returning NaN where all values are equal.

  ## Examples

      iex> x = Nx.tensor([[3, 5, 3], [2, 6, 1], [9, 3, 2], [1, 6, 8]])
      iex> Scholar.Stats.skew(x)
      #Nx.Tensor<
        f32[3]
        [0.9794093370437622, -0.8164965510368347, 0.9220733642578125]
      >
  """

  deftransform skew(tensor, opts \\ []) do
    skew_n(tensor, NimbleOptions.validate!(opts, @skew_schema))
  end

  defnp skew_n(tensor, opts \\ []) do
    m2 = moment(tensor, moment: 2, axis: opts[:axis], keep_axis: opts[:keep_axis])
    m3 = moment(tensor, moment: 3, axis: opts[:axis], keep_axis: opts[:keep_axis])
    m2_mod = m2 ** (3 / 2)

    if opts[:bias] do
      m3 / m2_mod
    else
      num_samples = Nx.axis_size(tensor, 0)
      m3 / m2_mod * Nx.sqrt(num_samples * (num_samples - 1)) / (num_samples - 2)
    end
  end

  @doc """
  Compute the kurtosis (Fisher or Pearson) of a dataset.

  ## Options

  #{NimbleOptions.docs(@kurtosis_schema)}

  ## Return Values

    The kurtosis of values along an axis, returning NaN where all values are equal.

  ## Examples

      iex> x = Nx.tensor([[3, 5, 3], [2, 6, 1], [9, 3, 2], [1, 6, 8]])
      iex> Scholar.Stats.kurtosis(x)
      #Nx.Tensor<
        f64[3]
        [-0.7980853277835589, -1.0, -0.839476813317479]
      >
  """
  deftransform kurtosis(tensor, opts \\ []) do
    kurtosis_n(tensor, NimbleOptions.validate!(opts, @kurtosis_schema))
  end

  defnp kurtosis_n(tensor, opts) do
    m2 = moment(tensor, moment: 2, axis: opts[:axis], keep_axis: opts[:keep_axis])
    m4 = moment(tensor, moment: 4, axis: opts[:axis], keep_axis: opts[:keep_axis])
    {_, num_bits} = Nx.type(tensor)

    m2_mask =
      Nx.select(m2 == 0, Nx.Constants.nan(if num_bits < 64, do: {:f, 32}, else: {:f, 64}), m2)

    vals = m4 / m2_mask ** 2
    num_samples = Nx.axis_size(tensor, 0)

    vals =
      cond do
        opts[:bias] or num_samples < 3 ->
          vals

        true ->
          1.0 / (num_samples - 2) / (num_samples - 3) *
            ((num_samples ** 2 - 1) * vals - 3 * (num_samples - 1) ** 2)
      end

    case opts[:variant] do
      :fisher -> vals - 3
      :pearson -> vals
    end
  end
end
