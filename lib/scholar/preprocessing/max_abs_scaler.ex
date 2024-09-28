defmodule Scholar.Preprocessing.MaxAbsScaler do
  @moduledoc """
  Scales a tensor by dividing each sample in batch by the maximum absolute value in the batch.

  Centering and scaling happen independently on each feature by computing the relevant
  statistics on the samples in the training set. The maximum absolute value is then
  stored to be used on new samples.
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:max_abs]}
  defstruct [:max_abs]

  opts_schema = [
    axes: [
      type: {:custom, Scholar.Options, :axes, []},
      doc: """
      Axes to calculate the max absolute value over. By default the absolute values
      are calculated between the whole tensors.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Compute the maximum absolute value of samples to be used for later scaling.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return values

    Returns a struct with the following parameters:

    * `max_abs`: the calculated maximum absolute value of samples.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> Scholar.Preprocessing.MaxAbsScaler.fit(t)
      %Scholar.Preprocessing.MaxAbsScaler{
        max_abs: Nx.tensor(
          [
            [2]
          ]
        )
      }
  """
  deftransform fit(tensor, opts \\ []) do
    fit_n(tensor, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(tensor, opts) do
    max_abs =
      Nx.abs(tensor)
      |> Nx.reduce_max(axes: opts[:axes], keep_axes: true)

    max_abs = Nx.select(max_abs == 0, 1, max_abs)

    %__MODULE__{max_abs: max_abs}
  end

  @doc """
  Performs the standardization of the tensor using a fitted scaler.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> scaler = Scholar.Preprocessing.MaxAbsScaler.fit(t)
      iex> Scholar.Preprocessing.MaxAbsScaler.transform(scaler, t)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.5, -0.5, 1.0],
          [1.0, 0.0, 0.0],
          [0.0, 0.5, -0.5]
        ]
      >
      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> scaler = Scholar.Preprocessing.MaxAbsScaler.fit(t)
      iex> new_tensor = Nx.tensor([[0.5, 1, -1], [0.3, 0.8, -1.6]])
      iex> Scholar.Preprocessing.MaxAbsScaler.transform(scaler, new_tensor)
      #Nx.Tensor<
        f32[2][3]
        [
          [0.25, 0.5, -0.5],
          [0.15000000596046448, 0.4000000059604645, -0.800000011920929]
        ]
      >
  """
  defn transform(%__MODULE__{max_abs: max_abs}, tensor) do
    tensor / max_abs
  end

  @doc """
  Standardizes the tensor by removing the mean and scaling to unit variance.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> Scholar.Preprocessing.MaxAbsScaler.fit_transform(t)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.5, -0.5, 1.0],
          [1.0, 0.0, 0.0],
          [0.0, 0.5, -0.5]
        ]
      >
  """
  defn fit_transform(tensor, opts \\ []) do
    tensor
    |> fit(opts)
    |> transform(tensor)
  end
end
