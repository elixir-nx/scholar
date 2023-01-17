defmodule Scholar.Options do
  # Useful NimbleOptions validations.
  @moduledoc false

  require Nx

  def axes(axes) do
    # Axes are further validated by Nx, including against the tensor.
    if axes == nil or is_list(axes) do
      {:ok, axes}
    else
      {:error, "expected :axes to be a list positive integers as axis"}
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

  def weights(weights) do
    if Nx.is_tensor(weights) or is_list(weights) do
      {:ok, weights}
    else
      {:error, "expected weights to be a tensor or a list, got: #{inspect(weights)}"}
    end
  end

  def positive_weights(weights) do
    if (Nx.is_tensor(weights) and Nx.to_number(Nx.all(Nx.greater(weights, 0.0))) == 1) or
         (is_list(weights) and Enum.all?(weights, fn x -> x > 0.0 end)) do
      {:ok, weights}
    else
      {:error,
       "expected weights to be a tensor or a list of positive numbers, got: #{inspect(weights)}"}
    end
  end

  def non_negative_weights(weights) do
    if (Nx.is_tensor(weights) and Nx.to_number(Nx.all(Nx.greater_equal(weights, 0.0))) == 1) or
         (is_list(weights) and Enum.all?(weights, fn x -> x >= 0.0 end)) do
      {:ok, weights}
    else
      {:error,
       "expected weights to be a tensor or a list of non-negative numbers, got: #{inspect(weights)}"}
    end
  end
end
