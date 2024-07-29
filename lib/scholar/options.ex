defmodule Scholar.Options do
  # Useful NimbleOptions validations.
  @moduledoc false

  require Nx

  def optimizer(value) do
    error =
      {:error,
       "expected :optimizer to be either a valid 0-arity function in Polaris.Optimizers or a valid {init_fn, update_fn} tuple"}

    case value do
      {init_fn, update_fn} when is_function(init_fn, 1) and is_function(update_fn, 3) ->
        {:ok, value}

      atom when is_atom(atom) ->
        mod = Polaris.Optimizers

        if Code.ensure_loaded(mod) == {:module, mod} and function_exported?(mod, atom, 0) do
          {:ok, atom}
        else
          error
        end

      _ ->
        error
    end
  end

  def axes(axes) do
    # Axes are further validated by Nx, including against the tensor.
    if axes == nil or is_list(axes) do
      {:ok, axes}
    else
      {:error, "expected :axes to be a list positive integers as axis"}
    end
  end

  def axis(axis) do
    # Axis is further validated by Nx, including against the tensor.
    if axis == nil or is_integer(axis) or is_atom(axis) do
      {:ok, axis}
    else
      {:error, "expected :axis to be an integers, atom or nil"}
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

  def non_negative_integer(num) do
    if is_integer(num) and num >= 0 do
      {:ok, num}
    else
      {:error, "expected a non-negative integer, got: #{inspect(num)}"}
    end
  end

  def weights(weights) do
    if is_nil(weights) or
         (Nx.is_tensor(weights) and Nx.rank(weights) in 0..1) or
         (is_list(weights) and Enum.all?(weights, &is_number/1)) do
      {:ok, weights}
    else
      {:error, "expected weights to be a flat tensor or a flat list, got: #{inspect(weights)}"}
    end
  end

  def multi_weights(weights) do
    if is_nil(weights) or
         (Nx.is_tensor(weights) and Nx.rank(weights) > 1) do
      {:ok, weights}
    else
      {:error,
       "expected weights to be a tensor with rank greater than 1, got: #{inspect(weights)}"}
    end
  end

  def key(key) do
    if Nx.is_tensor(key) and Nx.type(key) == {:u, 32} and Nx.shape(key) == {2} do
      {:ok, key}
    else
      {:error, "expected key to be a key (use Nx.Random.key/1), got: #{inspect(key)}"}
    end
  end

  def beta(beta) do
    if (is_number(beta) and beta >= 0) or (Nx.is_tensor(beta) and Nx.rank(beta) == 0) do
      {:ok, beta}
    else
      {:error, "expected 'beta' to be in the range [0, inf]"}
    end
  end
end
