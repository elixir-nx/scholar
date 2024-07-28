defmodule Scholar.Linear.LinearHelpers do
  require Nx
  import Nx.Defn
  import Scholar.Shared

  @moduledoc false

  defp valid_column_vector?(y, n_samples) do
    Nx.shape(y) == {n_samples, 1} and Nx.rank(y) == 2
  end

  @doc false
  def flatten_column_vector(y, n_samples) do
    is_column_vector? = valid_column_vector?(y, n_samples)

    if is_column_vector? do
      y |> Nx.flatten()
    else
      y
    end
  end

  @doc false
  def validate_y_shape(y, n_samples, module_name) do
    y = flatten_column_vector(y, n_samples)
    is_valid_target? = Nx.rank(y) == 1

    if not is_valid_target? do
      message =
        "#{inspect(module_name)} expected y to have shape {n_samples}, got tensor with shape: #{inspect(Nx.shape(y))}"

      raise ArgumentError, message
    else
      y
    end
  end

  @doc false
  def build_sample_weights(x, opts) do
    x_type = to_float_type(x)
    {num_samples, _} = Nx.shape(x)
    default_sample_weights = Nx.broadcast(Nx.as_type(1.0, x_type), {num_samples})
    {sample_weights, _} = Keyword.pop(opts, :sample_weights, default_sample_weights)

    # this is required for ridge regression
    sample_weights =
      if Nx.is_tensor(sample_weights),
        do: Nx.as_type(sample_weights, x_type),
        else: Nx.tensor(sample_weights, type: x_type)

    sample_weights
  end

  @doc false
  defn preprocess_data(x, y, sample_weights, opts) do
    if opts[:sample_weights_flag],
      do:
        {Nx.weighted_mean(x, sample_weights, axes: [0]),
         Nx.weighted_mean(y, sample_weights, axes: [0])},
      else: {Nx.mean(x, axes: [0]), Nx.mean(y, axes: [0])}
  end

  @doc false
  defn set_intercept(coeff, x_offset, y_offset, fit_intercept?) do
    if fit_intercept? do
      y_offset - Nx.dot(coeff, [-1], x_offset, [-1])
    else
      Nx.tensor(0.0, type: Nx.type(coeff))
    end
  end

  # Implements sample weighting by rescaling inputs and
  # targets by sqrt(sample_weight).
  @doc false
  defn rescale(x, y, sample_weights) do
    factor = Nx.sqrt(sample_weights)

    x_scaled =
      case Nx.shape(factor) do
        {} -> factor * x
        _ -> x * Nx.new_axis(factor, -1)
      end

    y_scaled =
      case Nx.rank(y) do
        1 -> factor * y
        _ -> y * Nx.new_axis(factor, -1)
      end

    {x_scaled, y_scaled}
  end
end
