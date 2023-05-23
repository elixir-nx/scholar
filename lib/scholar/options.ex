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
    if (Nx.is_tensor(weights) and Nx.rank(weights) == 1) or
         (is_list(weights) and Enum.all?(weights, &is_number/1)) do
      {:ok, weights}
    else
      {:error, "expected weights to be a flat tensor or a flat list, got: #{inspect(weights)}"}
    end
  end

  def seed(seed) do
    if is_integer(seed) or (Nx.is_tensor(seed) and Nx.type(seed) == {:u, 32} and Nx.shape(seed) == {2}) do
      {:ok, seed}
    else
      {:error, "expected seed to be an integer or key (use Nx.Random.key/1), got: #{inspect(seed)}"}
    end
  end

  def metric(:cosine), do: {:ok, :cosine}

  def metric({:minkowski, p}) when p == :infinity or (is_number(p) and p > 0),
    do: {:ok, {:minkowski, p}}

  def metric(metric) do
    {:error,
     "expected metric to be a :cosine or tuple {:minkowski, p} where p is a positive number or :infinity, got: #{inspect(metric)}"}
  end
end
