defmodule Scholar.Impute.SimpleImputer do
  @moduledoc """
  Univariate imputer for completing missing values with simple strategies.
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container, keep: [:missing_values], containers: [:statistics]}
  defstruct [:statistics, :missing_values]

  opts_schema = [
    missing_values: [
      type: {:or, [:float, :integer, {:in, [:nan]}]},
      default: :nan,
      doc: ~S"""
      The placeholder for the missing values. All occurrences of `:missing_values` will be imputed.
      """
    ],
    strategy: [
      type: {:in, [:mean, :median, :mode, :constant]},
      default: :mean,
      doc: ~S"""
      The imputation strategy.

      * `:mean` - replace missing values using the mean along each column.

      * `:median` - replace missing values using the median along each column.

      * `:mode` - replace missing using the most frequent value along each column.
        If there is more than one such value, only the smallest is returned.

      * `:constant` - replace missing values with `:fill_value`.
      """
    ],
    fill_value: [
      type: {:or, [:float, :integer]},
      default: 0.0,
      doc: ~S"""
      When strategy is set to `:constant`, `:fill_value` is used to replace all occurrences of `:missing_values`.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Univariate imputer for completing missing values with simple strategies.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:missing_values` - the same value as in `:missing_values`

    * `:statistics` - The imputation fill value for each feature. Computing statistics can result in
    [`Nx.Constant.nan/0`](https://hexdocs.pm/nx/Nx.Constants.html#nan/0) values.

  ## Examples

      iex> x = Nx.tensor([[1, 2, :nan], [3, 7, :nan], [:nan, 4, 5]])
      iex> Scholar.Impute.SimpleImputer.fit(x, strategy: :mean)
      %Scholar.Impute.SimpleImputer{
        statistics: Nx.tensor(
          [2.0, 4.333333492279053, 5.0]
        ),
        missing_values: :nan
      }
  """
  deftransform fit(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    input_rank = Nx.rank(x)

    if input_rank != 2 do
      raise ArgumentError, "Wrong input rank. Expected: 2, got: #{inspect(input_rank)}"
    end

    if opts[:missing_values] != :nan and
         Nx.any(Nx.is_nan(x)) == Nx.tensor(1, type: :u8) do
      raise ArgumentError,
            ":missing_values other than :nan possible only if there is no Nx.Constant.nan() in the array"
    end

    {type, _num_bits} = x_type = Nx.type(x)

    x =
      cond do
        opts[:strategy] == :constant and is_float(opts[:fill_value]) and
            type in [:s, :u] ->
          to_float(x)

        opts[:strategy] == :constant and is_integer(opts[:fill_value]) and
            type in [:f, :bf] ->
          {fill_value_type, _} = Nx.type(opts[:fill_value])

          raise ArgumentError,
                "Wrong type of `:fill_value` for the given data. Expected: :f or :bf, got: #{inspect(fill_value_type)}"

        true ->
          x
      end

    x =
      if opts[:missing_values] != :nan,
        do: Nx.select(Nx.equal(x, opts[:missing_values]), Nx.Constants.nan(), x),
        else: x

    {_num_rows, num_cols} = Nx.shape(x)

    statistics =
      cond do
        opts[:strategy] == :mean ->
          mean_op(x)

        opts[:strategy] == :median ->
          median_op(x)

        opts[:strategy] == :mode ->
          mode_op(x, type: x_type)

        true ->
          Nx.broadcast(opts[:fill_value], {num_cols})
      end

    missing_values = opts[:missing_values]
    %__MODULE__{statistics: statistics, missing_values: missing_values}
  end

  defnp mean_op(x) do
    mask = not Nx.is_nan(x)
    denominator = Nx.sum(mask, axes: [0])
    temp = Nx.select(mask, x, 0)
    numerator = Nx.sum(temp, axes: [0])

    Nx.select(
      denominator != 0,
      numerator / denominator,
      Nx.Constants.nan()
    )
  end

  defnp median_op(x) do
    axis = 0

    {num_rows, num_cols} = Nx.shape(x)
    x = Nx.sort(x, axis: axis)
    tensor = Nx.iota(Nx.shape(x), axis: axis) + Nx.is_nan(x) * num_rows
    res = Nx.argsort(tensor, axis: axis)
    x = Nx.take_along_axis(x, res, axis: axis)

    indices = Nx.broadcast(num_rows - 1, {num_cols})
    indices = indices - Nx.sum(Nx.is_nan(x), axes: [axis])
    half_indices = indices / 2
    floor = Nx.as_type(Nx.floor(half_indices), :s64) |> Nx.new_axis(axis)
    ceil = Nx.as_type(Nx.ceil(half_indices), :s64) |> Nx.new_axis(axis)
    nums1 = Nx.take_along_axis(x, floor, axis: axis)
    nums2 = Nx.take_along_axis(x, ceil, axis: axis)
    ((nums1 + nums2) / 2) |> Nx.squeeze()
  end

  defnp mode_op(x, opts) do
    type = opts[:type]

    axis = 0

    axis_length = Nx.axis_size(x, axis)
    x = Nx.sort(x, axis: 0)
    tensor = Nx.iota(Nx.shape(x), axis: axis) + Nx.is_nan(x) * axis_length
    res = Nx.argsort(tensor, axis: axis)
    sorted = Nx.take_along_axis(x, res, axis: axis)

    {num_rows, num_cols} = Nx.shape(tensor)

    num_elements = Nx.size(tensor)

    group_indices =
      Nx.concatenate(
        [
          Nx.broadcast(0, {1, num_cols}),
          Nx.slice_along_axis(sorted, 0, num_rows - 1, axis: axis) !=
            Nx.slice_along_axis(sorted, 1, num_rows - 1, axis: axis)
        ],
        axis: axis
      )
      |> Nx.cumulative_sum(axis: axis)

    counting_indices =
      [
        Nx.reshape(group_indices, {num_elements, 1}),
        Nx.shape(group_indices)
        |> Nx.iota(axis: 1)
        |> Nx.reshape({num_elements, 1})
      ]
      |> Nx.concatenate(axis: 1)

    largest_group_indices =
      Nx.broadcast(0, sorted)
      |> Nx.indexed_add(counting_indices, Nx.broadcast(1, {num_elements}))
      |> Nx.argmax(axis: axis, keep_axis: true)

    indices =
      largest_group_indices
      |> Nx.broadcast(group_indices)
      |> Nx.equal(group_indices)
      |> Nx.argmax(axis: axis, keep_axis: true)

    Nx.take_along_axis(sorted, indices, axis: axis) |> Nx.squeeze() |> Nx.as_type(type)
  end

  @doc """
  Impute all missing values in `x` using fitted imputer.

  ## Return Values

  The function returns input tensor with NaN replaced with values saved in fitted imputer.

  ## Examples

      iex> x = Nx.tensor([[1, 2, :nan], [3, 7, :nan], [:nan, 4, 5]])
      iex> imputer = Scholar.Impute.SimpleImputer.fit(x, strategy: :mean)
      iex> Scholar.Impute.SimpleImputer.transform(imputer, x)
      Nx.tensor(
        [
          [1.0, 2.0, 5.0],
          [3.0, 7.0, 5.0],
          [2.0, 4.0, 5.0]
        ]
      )

      iex> x = Nx.tensor([[1, 2, :nan], [3, 7, :nan], [:nan, 4, 5]])
      iex> y = Nx.tensor([[7, :nan, 6], [6, 9, :nan], [8, :nan, 1]])
      iex> imputer = Scholar.Impute.SimpleImputer.fit(x, strategy: :median)
      iex> Scholar.Impute.SimpleImputer.transform(imputer, y)
      Nx.tensor(
        [
          [7.0, 4.0, 6.0],
          [6.0, 9.0, 5.0],
          [8.0, 4.0, 1.0]
        ]
      )
  """
  deftransform transform(%__MODULE__{statistics: statistics, missing_values: missing_values}, x) do
    impute_values = Nx.new_axis(statistics, 0) |> Nx.broadcast(x)
    mask = if missing_values == :nan, do: Nx.is_nan(x), else: Nx.equal(x, missing_values)
    Nx.select(mask, impute_values, x)
  end
end
