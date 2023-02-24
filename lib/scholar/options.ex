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

  def metric(metric) do
    if metric == :cosine or
         (is_tuple(metric) and tuple_size(metric) == 2 and elem(metric, 0) == :minkowski and
            is_number(elem(metric, 1)) and
            elem(metric, 1) >= 0) do
      {:ok, metric}
    else
      {:error,
       "expected metric to be a :cosine or tuple {:minkowski, p} where p is non-negative number, got: #{inspect(metric)}"}
    end
  end
end
