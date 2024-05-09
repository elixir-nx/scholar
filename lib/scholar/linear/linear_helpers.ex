defmodule Scholar.Linear.LinearHelpers do
  require Nx
  import Nx.Defn

  @moduledoc false

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
      y_offset - Nx.dot(coeff, x_offset)
    else
      Nx.tensor(0.0, type: Nx.type(coeff))
    end
  end

  # Implements sample weighting by rescaling inputs and
  # targets by sqrt(sample_weight).
  @doc false
  defn rescale(x, y, sample_weights) do
    case Nx.shape(sample_weights) do
      {} = scalar ->
        scalar = Nx.sqrt(scalar)
        {scalar * x, scalar * y}

      _ ->
        scale = sample_weights |> Nx.sqrt() |> Nx.make_diagonal()
        {Nx.dot(scale, x), Nx.dot(scale, y)}
    end
  end

  # defn rescale(x, y, sample_weights) do
  #   factor = Nx.sqrt(sample_weights)

  #   x_scaled =
  #     case Nx.shape(factor) do
  #       {} -> factor * x
  #       _ -> Nx.new_axis(factor, 1) * x
  #     end

  #   {x_scaled, factor * y}
  # end
end
