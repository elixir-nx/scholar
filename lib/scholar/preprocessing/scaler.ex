defmodule Scholar.Preprocessing.Scaler do
  @moduledoc """
  The scaler struct and a set of functions for preprocessing data.
  """

  @enforce_keys [:data]
  defstruct [:data, :transformation]

  @type t :: %__MODULE__{}

  @doc """
  Standardizes the tensor by removing the mean and scaling to unit variance.

  Formula: `z = (x - u) / s`

  Where `u` is the mean of the samples, and `s` is the standard deviation.
  Standardization can be helpful in cases where the data follows a Gaussian distribution
  (or Normal distribution) without outliers.
  """
  @spec fit(tensor :: Nx.Tensor.t()) :: t()
  def fit(%Nx.Tensor{} = tensor) do
    %__MODULE__{
      data: tensor
    }
  end

  @spec fit(data :: list()) :: t() | list()
  def fit([_, _ | _] = data) do
    %__MODULE__{
      data: Nx.tensor(data)
    }
  end

  def fit([head | _] = data) when is_list(head) do
    %__MODULE__{
      data: Nx.tensor(data)
    }
  end

  def fit([_ | _] = data), do: data
  def fit([]), do: []

  def fit(data),
    do: raise(ArgumentError, "expected a list with a least two elements got #{inspect(data)}")

  @spec transform(scaler :: t()) :: t()
  def transform(%__MODULE__{data: data} = scaler) do
    tensor = Nx.to_tensor(data)

    transformation =
      tensor
      |> Nx.subtract(Nx.mean(tensor))
      |> Nx.divide(Nx.standard_deviation(tensor))

    %__MODULE__{
      scaler
      | transformation: transformation
    }
  end

  def transform(value),
    do: raise(ArgumentError, "expected a %Scholar.Preprocessing.Scaler{} got #{inspect(value)}")
end
