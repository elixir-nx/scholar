defmodule Scholar.Preprocessing.StandardScaler do
  import Nx.Defn

  defstruct [:deviation, :mean]

  opts_schema = [
    axes: [
      type: {:custom, Scholar.Options, :axes, []},
      doc: """
      Axes to calculate the distance over. By default the distance
      is calculated between the whole tensors.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  deftransform fit(tensor, opts \\ []) do
    NimbleOptions.validate!(opts, @opts_schema)
    {std, mean} = fit_n(tensor, opts)

    %__MODULE__{deviation: std, mean: mean}
  end

  defnp fit_n(tensor, opts) do
    std = Nx.standard_deviation(tensor, axes: opts[:axes], keep_axes: true)
    mean_reduced = Nx.mean(tensor, axes: opts[:axes], keep_axes: true)
    mean_reduced = Nx.select(Nx.equal(std, 0), 0.0, mean_reduced)

    {std, mean_reduced}
  end

  deftransform transform(%__MODULE__{deviation: std, mean: mean}, tensor) do
    scale(tensor, std, mean)
  end

  defn fit_transform(tensor, opts \\ []) do
    tensor
    |> fit(opts)
    |> transform(tensor)
  end

  defnp scale(tensor, std, mean) do
    (tensor - mean) / Nx.select(std == 0, 1.0, std)
  end
end
