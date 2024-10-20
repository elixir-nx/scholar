defmodule Scholar.Impute.KNNImputer do
  @moduledoc """
  Imputer for completing missing values using k-Nearest Neighbors.

  Each sample's missing values are imputed using the mean value from
    `n_neighbors` nearest neighbors found in the training set. Two samples are
    close if the features that neither is missing are close.
  """
  import Nx.Defn
  import Scholar.Metrics.Distance

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
    number_of_neighbors: [
      type: :pos_integer,
      default: 2,
      doc: "The number of nearest neighbors."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Imputer for completing missing values using k-Nearest Neighbors.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:missing_values` - the same value as in `:missing_values`

    * `:statistics` - The imputation fill value for each feature. Computing statistics can result in
    [`Nx.Constant.nan/0`](https://hexdocs.pm/nx/Nx.Constants.html#nan/0) values.

  ## Examples
  iex> x = Nx.tensor([[40.0, 2.0],[4.0, 5.0],[7.0, :nan],[:nan, 8.0],[11.0, 11.0]])
  iex> Scholar.Impute.KNNImputer.fit(x, number_of_neighbors: 2)
  %Scholar.Impute.KNNImputer{
  statistics: #Nx.Tensor<
        f32[5][2]
        [
            [NaN, NaN],
            [NaN, NaN],
            [NaN, 8.0],
            [7.5, NaN],
            [NaN, NaN]
          ]
  >,
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

    x =
      if opts[:missing_values] != :nan,
        do: Nx.select(Nx.equal(x, opts[:missing_values]), Nx.Constants.nan(), x),
        else: x

    num_neighbors = opts[:number_of_neighbors]

    if num_neighbors < 1 do
      raise ArgumentError, "Number of neighbors must be greater than 0"
    end

    {rows, cols} = Nx.shape(x)

    row_nan_count = Nx.sum(Nx.is_nan(x), axes: [1])
    # row with only 1 non nan value is also considered as all nan row
    all_nan_rows =
      Nx.select(Nx.greater_equal(row_nan_count, cols - 1), Nx.tensor(1), Nx.tensor(0))

    all_nan_rows_count = Nx.sum(all_nan_rows)

    if num_neighbors > rows - 1 - Nx.to_number(all_nan_rows_count) do
      raise ArgumentError,
            "Number of neighbors rows must be less than number valid of rows - 1 (valid row is row with more than 1 non nan value)"
    end

    placeholder_value = Nx.Constants.nan() |> Nx.tensor()

    statistics = knn_impute(x, placeholder_value, num_neighbors: num_neighbors)
    #     statistics = all_nan_rows_count
    missing_values = opts[:missing_values]
    %__MODULE__{statistics: statistics, missing_values: missing_values}
  end

  @doc """
  Impute all missing values in `x` using fitted imputer.

  ## Return Values

  The function returns input tensor with NaN replaced with values saved in fitted imputer.

  ## Examples

      iex> x = Nx.tensor([[40.0, 2.0],[4.0, 5.0],[7.0, :nan],[:nan, 8.0],[11.0, 11.0]])
      iex> imputer = Scholar.Impute.KNNImputer.fit(x, number_of_neighbors: 2)
      iex> Scholar.Impute.KNNImputer.transform(imputer, x)
      Nx.tensor(
        [
          [40.0, 2.0],
          [4.0, 5.0],
          [7.0, 8.0],
          [7.5, 8.0],
          [11.0, 11.0]
        ]
      )
  """
  deftransform transform(%__MODULE__{statistics: statistics, missing_values: missing_values}, x) do
    mask = if missing_values == :nan, do: Nx.is_nan(x), else: Nx.equal(x, missing_values)
    Nx.select(mask, statistics, x)
  end

  defnp knn_impute(x, placeholder_value, opts \\ []) do
    mask = Nx.is_nan(x)
    {num_rows, num_cols} = Nx.shape(x)
    num_neighbors = opts[:num_neighbors]

    values_to_impute = Nx.broadcast(placeholder_value, x)

    {_, values_to_impute} =
      while {{row = 0, mask, num_neighbors, num_rows, x}, values_to_impute},
            Nx.less(row, num_rows) do
        {_, values_to_impute} =
          while {{col = 0, mask, num_neighbors, num_cols, row, x}, values_to_impute},
                Nx.less(col, num_cols) do
            if mask[row][col] > 0 do
              {rows, cols} = Nx.shape(x)

              neighbor_avg =
                calculate_knn(x, row, col, rows: rows, num_neighbors: opts[:num_neighbors])

              indices =
                [Nx.stack(row), Nx.stack(col)]
                |> Nx.concatenate()
                |> Nx.stack()

              values_to_impute = Nx.indexed_put(values_to_impute, indices, Nx.stack(neighbor_avg))
              {{col + 1, mask, num_neighbors, cols, row, x}, values_to_impute}
            else
              {{col + 1, mask, num_neighbors, num_cols, row, x}, values_to_impute}
            end
          end

        {{row + 1, mask, num_neighbors, num_rows, x}, values_to_impute}
      end

    values_to_impute
  end

  defnp calculate_knn(x, nan_row, nan_col, opts \\ []) do
    opts = keyword!(opts, rows: 1, num_neighbors: 2)
    rows = opts[:rows]
    num_neighbors = opts[:num_neighbors]

    row_distances = Nx.iota({rows}, type: {:f, 32})

    row_with_value_to_fill = x[nan_row]

    # calculate distance between row with nan to fill and all other rows where distance
    # to the row is under its index in the tensor
    {_, row_distances} =
      while {{i = 0, x, row_with_value_to_fill, rows, nan_row, nan_col}, row_distances},
            Nx.less(i, rows) do
        potential_donor = x[i]

        if i == nan_row do
          distance = Nx.Constants.infinity({:f, 32})
          row_distances = Nx.indexed_put(row_distances, Nx.new_axis(i, 0), distance)
          {{i + 1, x, row_with_value_to_fill, rows, nan_row, nan_col}, row_distances}
        else
          distance = nan_euclidian(row_with_value_to_fill, nan_col, potential_donor)
          row_distances = Nx.indexed_put(row_distances, Nx.new_axis(i, 0), distance)
          {{i + 1, x, row_with_value_to_fill, rows, nan_row, nan_col}, row_distances}
        end
      end

    {_, indices} = Nx.top_k(-row_distances, k: num_neighbors)

    gather_indices = Nx.stack([indices, Nx.broadcast(nan_col, indices)], axis: 1)
    values = Nx.gather(x, gather_indices)
    Nx.sum(values) / num_neighbors
  end

  # nan_col is the column of the value to impute
  defnp nan_euclidian(row, nan_col, potential_neighbor) do
    {coordinates} = Nx.shape(row)

    # minus nan column
    coordinates = coordinates - 1

    # inputes zeros in nan_col to calculate distance with squared_euclidean
    new_row = Nx.indexed_put(row, Nx.new_axis(nan_col, 0), Nx.tensor(0))

    # if potential neighbor has nan in nan_col, we don't want to calculate distance and the case if potential_neighbour is the row to impute
    {potential_neighbor} =
      if potential_neighbor[nan_col] == Nx.Constants.nan() do
        potential_neighbor = Nx.broadcast(Nx.Constants.infinity({:f, 32}), potential_neighbor)
        {potential_neighbor}
      else
        # inputes zeros in nan_col to calculate distance with squared_euclidean - distance will be 0 so no change to the distance value
        potential_neighbor =
          Nx.indexed_put(potential_neighbor, Nx.new_axis(nan_col, 0), Nx.tensor(0))

        {potential_neighbor}
      end

    # calculates how many values are present in the row without nan_col to calculate weight for the distance
    present_coordinates = Nx.sum(Nx.logical_not(Nx.is_nan(potential_neighbor))) - 1

    # if row has all nans we skip it
    {weight, potential_neighbor} =
      if present_coordinates == 0 do
        potential_neighbor = Nx.broadcast(Nx.Constants.infinity({:f, 32}), potential_neighbor)
        weight = 0
        {weight, potential_neighbor}
      else
        potential_neighbor = Nx.select(Nx.is_nan(potential_neighbor), new_row, potential_neighbor)
        weight = coordinates / present_coordinates
        {weight, potential_neighbor}
      end

    # calculating weighted euclidian distance
    distance = Nx.sqrt(weight * squared_euclidean(new_row, potential_neighbor))

    # return inf if potential_row is row to impute
    Nx.select(Nx.is_nan(distance), Nx.Constants.infinity({:f, 32}), distance)
  end
end
