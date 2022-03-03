defmodule Scholar.Preprocessing do
  @moduledoc """
  Set of functions for preprocessing data.
  """

  import Nx.Defn

  @doc """
  Standardizes the tensor by removing the mean and scaling to unit variance.

  Formula: `z = (x - u) / s`

  Where `u` is the mean of the samples, and `s` is the standard deviation.
  Standardization can be helpful in cases where the data follows a Gaussian distribution
  (or Normal distribution) without outliers.

  ## Examples

        iex> Scholar.Preprocessing.standard_scaler(Nx.tensor([1,2,3]))
        #Nx.Tensor<
          f32[3]
          [-1.2247447967529297, 0.0, 1.2247447967529297]
        >

        iex> Scholar.Preprocessing.standard_scaler(Nx.tensor([[1, -1, 2], [2, 0, 0], [0, 1, -1]]))
        #Nx.Tensor<
          f32[3][3]
          [
            [0.5212860703468323, -1.3553436994552612, 1.4596009254455566],
            [1.4596009254455566, -0.4170288145542145, -0.4170288145542145],
            [-0.4170288145542145, 0.5212860703468323, -1.3553436994552612]
          ]
        >

        iex> Scholar.Preprocessing.standard_scaler(42)
        #Nx.Tensor<
          f32
          42
        >
  """
  @spec standard_scaler(tensor :: Nx.Tensor.t()) :: Nx.Tensor.t()
  defn standard_scaler(tensor) do
    tensor = Nx.to_tensor(tensor)
    std = Nx.standard_deviation(tensor)

    if std == Nx.tensor(0.0) do
      tensor
    else
      tensor
      |> Nx.subtract(Nx.mean(tensor))
      |> Nx.divide(std)
    end
  end
end
