defmodule Scholar.Preprocessing.RobustScaler do
  @moduledoc ~S"""
  Scale features using statistics that are robust to outliers.

  This Scaler removes the median and scales the data according to
  the quantile range (defaults to IQR: Interquartile Range).
  The IQR is the range between the 1st quartile (25th quantile)
  and the 3rd quartile (75th quantile).
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:medians, :iqr]}
  defstruct [:medians, :iqr]

  opts_schema = [
    quantile_range: [
      type: {:custom, Scholar.Options, :quantile_range, []},
      default: {25.0, 75.0},
      doc: """
      Quantile range as a tuple {q_min, q_max} defining the range of quantiles
      to include. Must satisfy 0.0 < q_min < q_max < 100.0.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Compute the median and quantiles to be used for scaling.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return values

  Returns a struct with the following parameters:

  * `:iqr` - the calculated interquartile range.

  * `:medians` - the calculated medians of each feature across samples.

  ## Examples

      iex> Scholar.Preprocessing.RobustScaler.fit(Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]]))
      %Scholar.Preprocessing.RobustScaler{
        medians: Nx.tensor([1, 0, 0]),
        iqr: Nx.tensor([1.0, 1.0, 1.5])
      }
  """
  deftransform fit(tensor, opts \\ []) do
    fit_n(tensor, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(tensor, opts) do
    check_for_rank(tensor)

    {q_min, q_max} = opts[:quantile_range]

    medians = Nx.median(tensor, axis: 0)

    sorted_tensor = Nx.sort(tensor, axis: 0)

    q_min = percentile(sorted_tensor, q_min)
    q_max = percentile(sorted_tensor, q_max)

    iqr = q_max - q_min

    %__MODULE__{medians: medians, iqr: iqr}
  end

  @doc """
  Performs centering and scaling of the tensor using a fitted scaler.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> scaler = Scholar.Preprocessing.RobustScaler.fit(t)
      %Scholar.Preprocessing.RobustScaler{
        medians: Nx.tensor([1, 0, 0]),
        iqr: Nx.tensor([1.0, 1.0, 1.5])
      }
      iex> Scholar.Preprocessing.RobustScaler.transform(scaler, t)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, -1.0, 1.3333333730697632],
          [1.0, 0.0, 0.0],
          [-1.0, 1.0, -0.6666666865348816]
        ]
      >
  """
  defn transform(%__MODULE__{medians: medians, iqr: iqr}, tensor) do
    check_for_rank(tensor)
    scale(tensor, medians, iqr)
  end

  @doc """
  Computes the scaling parameters and applies them to transform the tensor.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> Scholar.Preprocessing.RobustScaler.fit_transform(t)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, -1.0, 1.3333333730697632],
          [1.0, 0.0, 0.0],
          [-1.0, 1.0, -0.6666666865348816]
        ]
      >
  """
  defn fit_transform(tensor, opts \\ []) do
    tensor
    |> fit(opts)
    |> transform(tensor)
  end

  defnp scale(tensor, medians, iqr) do
    (tensor - medians) / Nx.select(iqr == 0, 1.0, iqr)
  end

  defnp percentile(sorted_tensor, p) do
    num_rows = Nx.axis_size(sorted_tensor, 0)
    idx = p / 100 * (num_rows - 1)

    lower_idx = Nx.floor(idx) |> Nx.as_type(:s64)
    upper_idx = Nx.ceil(idx) |> Nx.as_type(:s64)

    lower_values = Nx.take(sorted_tensor, lower_idx, axis: 0)
    upper_values = Nx.take(sorted_tensor, upper_idx, axis: 0)

    weight_upper = idx - Nx.floor(idx)
    weight_lower = 1.0 - weight_upper
    lower_values * weight_lower + upper_values * weight_upper
  end

  defnp check_for_rank(tensor) do
    if Nx.rank(tensor) != 2 do
      raise ArgumentError,
            """
            expected tensor to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(tensor))}\
            """
    end
  end
end
