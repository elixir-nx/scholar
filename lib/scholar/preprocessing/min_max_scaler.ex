defmodule Scholar.Preprocessing.MinMaxScaler do
  @moduledoc """
  Scales a tensor by dividing each sample in batch by maximum absolute value in the batch

  Centering and scaling happen independently on each feature by computing the relevant
  statistics on the samples in the training set. Maximum absolute value then is
  stored to be used on new samples.
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:min_data, :max_data, :min_bound, :max_bound]}
  defstruct [:min_data, :max_data, :min_bound, :max_bound]

  opts_schema = [
    axes: [
      type: {:custom, Scholar.Options, :axes, []},
      doc: """
      Axes to calculate the max absolute value over. By default the absolute values
      are calculated between the whole tensors.
      """
    ],
    min_bound: [
      type: {:or, [:integer, :float]},
      default: 0,
      doc: """
      The lower boundary of the desired range of transformed data.
      """
    ],
    max_bound: [
      type: {:or, [:integer, :float]},
      default: 1,
      doc: """
      The upper boundary of the desired range of transformed data.
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

    * `min_data`: the calculated minimum value of samples.

    * `max_data`: the calculated maximum value of samples.

    * `min_bound`: The lower boundary of the desired range of transformed data.

    * `max_bound`: The upper boundary of the desired range of transformed data.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> Scholar.Preprocessing.MinMaxScaler.fit(t)
      %Scholar.Preprocessing.MinMaxScaler{
        min_data: Nx.tensor(
          [
            [-1]
          ]
        ),
        max_data: Nx.tensor(
          [
            [2]
          ]
        ),
        min_bound: Nx.tensor(
          0
        ),
        max_bound: Nx.tensor(
          1
        )
      }
  """
  deftransform fit(tensor, opts \\ []) do
    fit_n(tensor, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(tensor, opts) do
    if opts[:max_bound] <= opts[:min_bound] do
      raise ArgumentError,
            "expected :max to be greater than :min"
    else
      reduced_max = Nx.reduce_max(tensor, axes: opts[:axes], keep_axes: true)
      reduced_min = Nx.reduce_min(tensor, axes: opts[:axes], keep_axes: true)

      %__MODULE__{
        min_data: reduced_min,
        max_data: reduced_max,
        min_bound: opts[:min_bound],
        max_bound: opts[:max_bound]
      }
    end
  end

  @doc """
  Performs the standardization of the tensor using a fitted scaler.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> scaler = Scholar.Preprocessing.MinMaxScaler.fit(t)
      iex> Scholar.Preprocessing.MinMaxScaler.transform(scaler, t)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.6666666865348816, 0.0, 1.0],
          [1.0, 0.3333333432674408, 0.3333333432674408],
          [0.3333333432674408, 0.6666666865348816, 0.0]
        ]
      >

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> scaler = Scholar.Preprocessing.MinMaxScaler.fit(t)
      iex> new_tensor = Nx.tensor([[0.5, 1, -1], [0.3, 0.8, -1.6]])
      iex> Scholar.Preprocessing.MinMaxScaler.transform(scaler, new_tensor)
      #Nx.Tensor<
        f32[2][3]
        [
          [0.5, 0.6666666865348816, 0.0],
          [0.43333330750465393, 0.5999999642372131, -0.20000000298023224]
        ]
      >
  """
  defn transform(
         %__MODULE__{
           min_data: min_data,
           max_data: max_data,
           min_bound: min_bound,
           max_bound: max_bound
         },
         tensor
       ) do
    denominator = max_data - min_data
    denominator = Nx.select(denominator == 0, 1, denominator)
    x_std = (tensor - min_data) / denominator
    x_std * (max_bound - min_bound) + min_bound
  end

  @doc """
  Standardizes the tensor by removing the mean and scaling to unit variance.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> Scholar.Preprocessing.MinMaxScaler.fit_transform(t)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.6666666865348816, 0.0, 1.0],
          [1.0, 0.3333333432674408, 0.3333333432674408],
          [0.3333333432674408, 0.6666666865348816, 0.0]
        ]
      >
  """
  defn fit_transform(tensor, opts \\ []) do
    tensor
    |> fit(opts)
    |> transform(tensor)
  end
end
