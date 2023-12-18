defmodule Scholar.Preprocessing.StandardScaler do
  @moduledoc ~S"""
  Standardizes the tensor by removing the mean and scaling to unit variance.

  Formula for input tensor $x$:

  $$
  z = \frac{x - \mu}{\sigma}
  $$

  Where $\mu$ is the mean of the samples, and $\sigma$ is the standard deviation.
  Standardization can be helpful in cases where the data follows
  a Gaussian distribution (or Normal distribution) without outliers.

  Centering and scaling happen independently on each feature by computing the relevant
  statistics on the samples in the training set. Mean and standard deviation are then
  stored to be used on new samples.
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:standard_deviation, :mean]}
  defstruct [:standard_deviation, :mean]

  opts_schema = [
    axes: [
      type: {:custom, Scholar.Options, :axes, []},
      doc: """
      Axes to calculate the distance over. By default the distance
      is calculated between the whole tensors.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
  Compute the standard deviation and mean of samples to be used for later scaling.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return values

  Returns a struct with the following parameters:

    * `standard_deviation`: the calculated standard deviation of samples.

    * `mean`: the calculated mean of samples.

  ## Examples

      iex> Scholar.Preprocessing.StandardScaler.fit(Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]]))
      %Scholar.Preprocessing.StandardScaler{
        standard_deviation: Nx.tensor(
          [
            [1.0657403469085693]
          ]
        ),
        mean: Nx.tensor(
          [
            [0.4444444477558136]
          ]
        )
      }
  """
  deftransform fit(tensor, opts \\ []) do
    NimbleOptions.validate!(opts, @opts_schema)
    fit_n(tensor, opts)
  end

  defnp fit_n(tensor, opts) do
    std = Nx.standard_deviation(tensor, axes: opts[:axes], keep_axes: true)
    mean_reduced = Nx.mean(tensor, axes: opts[:axes], keep_axes: true)
    mean_reduced = Nx.select(std == 0, 0.0, mean_reduced)
    %__MODULE__{standard_deviation: std, mean: mean_reduced}
  end

  @doc """
  Performs the standardization of the tensor using a fitted scaler.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> scaler = Scholar.Preprocessing.StandardScaler.fit(t)
      %Scholar.Preprocessing.StandardScaler{
        standard_deviation: Nx.tensor(
          [
            [1.0657403469085693]
          ]
        ),
        mean: Nx.tensor(
          [
            [0.4444444477558136]
          ]
        )
      }
      iex> Scholar.Preprocessing.StandardScaler.transform(scaler, t)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.5212860703468323, -1.3553436994552612, 1.4596009254455566],
          [1.4596009254455566, -0.4170288145542145, -0.4170288145542145],
          [-0.4170288145542145, 0.5212860703468323, -1.3553436994552612]
        ]
      >
  """
  defn transform(%__MODULE__{standard_deviation: std, mean: mean}, tensor) do
    scale(tensor, std, mean)
  end

  @doc """
  Standardizes the tensor by removing the mean and scaling to unit variance.

  ## Examples

      iex> t = Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
      iex> Scholar.Preprocessing.StandardScaler.fit_transform(t)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.5212860703468323, -1.3553436994552612, 1.4596009254455566],
          [1.4596009254455566, -0.4170288145542145, -0.4170288145542145],
          [-0.4170288145542145, 0.5212860703468323, -1.3553436994552612]
        ]
      >
  """
  defn fit_transform(tensor, opts \\ []) do
    tensor
    |> fit(opts)
    |> transform(tensor)
  end

  defnp scale(tensor, std, mean) do
    (tensor - mean) / Nx.select(std == 0, 1.0, std)
  end
end
