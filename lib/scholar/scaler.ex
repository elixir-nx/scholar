defmodule Scholar.Scaler do
  @moduledoc """
  Set of functions to change data scale.
  """

  @doc """
  Standardizes the tensor by removing the mean and scaling to unit variance.
  Formula: `z = (x - u) / s`
  Where `u` is the mean of the samples, and `s` is the standard deviation.
  Standardization can be helpful in cases where the data follows a Gaussian distribution
  (or Normal distribution) without outliers.
  """
  @spec standard_scaler(tensor :: Nx.Tensor.t()) :: Nx.Tensor.t()
  def standard_scaler(tensor) do
    tensor
    |> Nx.to_tensor()
    |> Nx.subtract(Nx.mean(tensor))
    |> Nx.divide(Nx.standard_deviation(tensor))
  end
end
