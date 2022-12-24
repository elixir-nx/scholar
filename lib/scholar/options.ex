defmodule Scholar.Options do
  # Useful NimbleOptions validations.
  @moduledoc false

  def axes(axes) do
    # Axes are further validated by Nx, including against the tensor.
    if axes == nil or is_list(axes) do
      {:ok, axes}
    else
      {:error, "expected :axes to be a list positive integers as axis"}
    end
  end

  def weights(weights) do
    import Nx, only: [is_tensor: 1, tensor: 1]
    # weights are further validated by Nx, including against the tensor.
    cond do
      weights == nil or is_tensor(weights) ->
        {:ok, weights}
      is_list(weights) ->
        {:ok, tensor(weights)}
      true ->
        {:error, "expected :weights to be a list or a tensor"}
    end
  end

  def type(type) do
    {:ok, Nx.Type.normalize!(type)}
  end

  def positive_number(num) do
    if is_number(num) and num > 0 do
      {:ok, num}
    else
      {:error, "expected positive number, got: #{inspect(num)}"}
    end
  end

  def non_negative_number(num) do
    if is_number(num) and num >= 0 do
      {:ok, num}
    else
      {:error, "expected a non-negative number, got: #{inspect(num)}"}
    end
  end
end
