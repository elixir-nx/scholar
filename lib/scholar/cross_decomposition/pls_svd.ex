defmodule Scholar.CrossDecomposition.PLSSVD do
  @moduledoc """

  """
  import Nx.Defn
  import Scholar.Shared

  opts_schema = [
    num_components: [
      default: 2,
      type: :pos_integer,
      doc: "The number of components to keep. Should be in `[1,
        min(n_samples, n_features, n_targets)]`."
    ],
    scale: [
      default: true,
      type: :boolean,
      doc: "Whether to scale `x` and `x`."
    ],
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values


  ## Examples

  """

  deftransform fit(x, y opts \\ []) do
    fit_n(x, y,  NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(x, y, opts) do
    {x, y} = check_x_y(x, y, opts)
    num_components = opts[:num_components]
    {x, y, x_mean, y_mean, x_std, y_std} = center_scale_xy(x, y, opts)
  end

  defnp check_x_y(x, y, opts) do
    y =
      case Nx.shape(y) do
        {n} -> Nx.reshape(y, {n, 1})
        _ -> y
      end

    num_components = opts[:num_components]
    {num_samples, num_features} = Nx.shape(x)
    {num_samples_y, num_targets} = Nx.axis_shape(y, 0)

    cond do
      num_samples != num_samples_y ->
        raise ArgumentError,
              """
              num_samples must be the same for x and y \
              x num_samples = #{num_samples}, y num_samples = #{num_samples_y}
              """

      num_components > num_features ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_features = #{num_features}, got #{num_components}
              """

      num_components > num_samples ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_samples = #{num_samples}, got #{num_components}
              """

      num_components > num_targets ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_targets = #{num_targets}, got #{num_components}
              """

      true ->
        nil
    end
    {x, y}
  end

  defnp center_scale_x_y(x, y, opts) do
    scale = opts[:scale]
    x_mean = Nx.mean(x, axis: 0)
    x = x - x_mean

    y_mean = Nx.mean(y, axis: 0)
    y = y - y_mean

    if scale do 
      x_std = Nx.standard_deviation(x, axes: [0], ddof: 1)
      x_std = Nx.select(x_std == 0.0, 1.0, x_std)
      x = x / Nx.broadcast(x_std, Nx.shape(x))

      y_std = Nx.standard_deviation(y, axes: [0], ddof: 1)
      y_std = Nx.select(y_std == 0.0, 1.0, y_std)
      y = y / Nx.broadcast(y_std, Nx.shape(y))

      {x, y, x_mean, y_mean, x_std, y_std}
    else
      x_std = Nx.broadcast(1, {Nx.axis_size(x, 1)})
      y_std = Nx.broadcast(1, {Nx.axis_size(y, 1)})
      {x, y, x_mean, y_mean, x_std, y_std}
    end
  end
end
