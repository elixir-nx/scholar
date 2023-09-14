defmodule Scholar.Scaler.StandardScaler do
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

    std = Nx.standard_deviation(tensor, axes: opts[:axes], keep_axes: true)
    mean_reduced = Nx.mean(tensor, axes: opts[:axes], keep_axes: true)
    mean_reduced = Nx.select(Nx.equal(std, 0), 0.0, mean_reduced)
    %__MODULE__{deviation: std, mean: mean_reduced}
  end

  deftransform transform(tensor, %__MODULE__{deviation: std, mean: mean}) do
    scale(tensor, std, mean)
  end

  deftransform fit_transform(tensor, opts \\ []) do
    scaler = __MODULE__.fit(tensor, opts)
    __MODULE__.transform(tensor, scaler)
  end

  defnp scale(tensor, std, mean) do
    (tensor - mean) / Nx.select(std == 0, 1.0, std)
  end
end
