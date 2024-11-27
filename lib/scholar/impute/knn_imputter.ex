defmodule Scholar.Impute.KNNImputter do
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
      type: {:or, [:float, :integer, {:in, [:infinity, :neg_infinity, :nan]}]},
      default: :nan,
      doc: ~S"""
      The placeholder for the missing values. All occurrences of `:missing_values` will be imputed.

      The default value expects there are no NaNs in the input tensor.
      """
    ],
    num_neighbors: [
      type: :pos_integer,
      default: 2,
      doc: "The number of nearest neighbors."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Imputter for completing missing values using k-Nearest Neighbors.

  Preconditions:
    *  The number of neighbors must be less than the number of valid rows - 1.
    *  A valid row is a row with more than 1 non-NaN values. Otherwise it is better to use a simpler imputter.
    *  When you set a value different than :nan in `missing_values` there should be no NaNs in the input tensor

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:missing_values` - the same value as in the `:missing_values` option

    * `:statistics` - The imputation fill value for each feature. Computing statistics can result in values.

  ## Examples

      iex> x = Nx.tensor([[40.0, 2.0],[4.0, 5.0],[7.0, :nan],[:nan, 8.0],[11.0, 11.0]])
      iex> Scholar.Impute.KNNImputter.fit(x, num_neighbors: 2)
      %Scholar.Impute.KNNImputter{
        statistics: Nx.tensor(
          [
                  [:nan, :nan],
                  [:nan, :nan],
                  [:nan, 8.0],
                  [7.5, :nan],
                  [:nan, :nan]
                ]
        ),
        missing_values: :nan
      }

  """

  deftransform fit(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    input_rank = Nx.rank(x)

    if input_rank != 2 do
      raise ArgumentError, "wrong input rank. Expected: 2, got: #{inspect(input_rank)}"
    end

    missing_values = opts[:missing_values]

    x =
      if missing_values != :nan,
         do: Nx.select(Nx.equal(x, missing_values), :nan, x),
         else: x


    statistics = knn_impute(x, num_neighbors: opts[:num_neighbors], missing_values: missing_values)
    %__MODULE__{statistics: statistics, missing_values: missing_values}
  end

  @doc """
  Impute all missing values in `x` using fitted imputer.

  ## Return Values

  The function returns input tensor with NaN replaced with values saved in fitted imputer.

  ## Examples

      iex> x = Nx.tensor([[40.0, 2.0],[4.0, 5.0],[7.0, :nan],[:nan, 8.0],[11.0, 11.0]])
      iex> imputer = Scholar.Impute.KNNImputter.fit(x, num_neighbors: 2)
      iex> Scholar.Impute.KNNImputter.transform(imputer, x)
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

  defnp knn_impute(x, opts \\ []) do
    mask = Nx.is_nan(x)
    {num_rows, num_cols} = Nx.shape(x)
    num_neighbors = opts[:num_neighbors]

    placeholder_value = Nx.tensor(:nan)

    values_to_impute = Nx.broadcast(placeholder_value, x)

    {_, values_to_impute} =
      while {{row = 0, mask, num_neighbors, num_rows, x}, values_to_impute},
            row < num_rows do
        {_, values_to_impute} =
          while {{col = 0, mask, num_neighbors, num_cols, row, x}, values_to_impute},
                col < num_cols do
            if mask[row][col] do
              {rows, cols} = Nx.shape(x)

              neighbor_avg =
                calculate_knn(x, row, col, rows: rows, num_neighbors: opts[:num_neighbors])

              values_to_impute = Nx.put_slice(values_to_impute, [row, col], Nx.reshape(neighbor_avg, {1, 1}))
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
#    {_, row_distances} =
#      while {{i = 0, x, row_with_value_to_fill, rows, nan_row, nan_col}, row_distances},
#            i < rows do
#
#        potential_donor = x[i]
#
#        distance =
#          if i == nan_row do
#            Nx.Constants.infinity(Nx.type(row_with_value_to_fill))
#          else
#            nan_euclidean(row_with_value_to_fill, nan_col, potential_donor)
#          end
#
#        row_distances = Nx.indexed_put(row_distances, Nx.new_axis(i, 0), distance)
#        {{i + 1, x, row_with_value_to_fill, rows, nan_row, nan_col}, row_distances}
#      end
    potential_donors = Nx.vectorize(x, :rows)
    distances = nan_euclidean(row_with_value_to_fill, nan_col, potential_donors) |> Nx.devectorize()
    row_distances = Nx.indexed_put(distances, Nx.tensor(nan_row), Nx.Constants.infinity())

    {_, indices} = Nx.top_k(-row_distances, k: num_neighbors)

    gather_indices = Nx.stack([indices, Nx.broadcast(nan_col, indices)], axis: 1)
    values = Nx.gather(x, gather_indices)
    Nx.sum(values) / num_neighbors
  end

  # nan_col is the column of the value to impute
  defnp nan_euclidean(row, nan_col, potential_neighbor) do
    {coordinates} = Nx.shape(row)

    # minus nan column
    coordinates = coordinates - 1

    # inputes zeros in nan_col to calculate distance with squared_euclidean
    new_row = Nx.indexed_put(row, Nx.new_axis(nan_col, 0), Nx.tensor(0))

    # if potential neighbor has nan in nan_col, we don't want to calculate distance and the case if potential_neighbour is the row to impute
    {potential_neighbor} =
      if Nx.is_nan(potential_neighbor[nan_col]) do
        potential_neighbor =
          Nx.broadcast(Nx.Constants.infinity(Nx.type(potential_neighbor)), potential_neighbor)

        {potential_neighbor}
      else
        # inputes zeros in nan_col to calculate distance with squared_euclidean - distance will be 0 so no change to the distance value
        potential_neighbor =
          Nx.indexed_put(
            potential_neighbor,
            Nx.new_axis(nan_col, 0),
            Nx.tensor(0, type: Nx.type(row))
          )

        {potential_neighbor}
      end

    # calculates how many values are present in the row without nan_col to calculate weight for the distance
    present_coordinates = Nx.sum(Nx.logical_not(Nx.is_nan(potential_neighbor))) - 1

    # if row has all nans we skip it
    {weight, potential_neighbor} =
      if present_coordinates == 0 do
        potential_neighbor =
          Nx.broadcast(Nx.Constants.infinity(Nx.type(potential_neighbor)), potential_neighbor)

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
    Nx.select(Nx.is_nan(distance), Nx.Constants.infinity(Nx.type(distance)), distance)
  end
end
