defmodule Scholar.Preprocessing.Normalizer do
  @moduledoc """
  Implements functionality for rescaling tensor to unit norm. It enables to apply normalization along any combination of axes.
  """
  import Nx.Defn
  import Scholar.Shared

  normalize_schema = [
    axes: [
      type: {:custom, Scholar.Options, :axes, []},
      doc: """
      Axes to calculate the distance over. By default the distance
      is calculated between the whole tensors.
      """
    ],
    norm: [
      type: {:in, [:euclidean, :chebyshev, :manhattan]},
      default: :euclidean,
      doc: """
      The norm to use to normalize each non zero sample.
      Possible options are `:euclidean`, `:manhattan`, and `:chebyshev`
      """
    ]
  ]

  @normalize_schema NimbleOptions.new!(normalize_schema)

  @doc """
  Normalize samples individually to unit norm.

  The zero-tensors cannot be normalized and they stay the same
  after normalization.

  ## Options

  #{NimbleOptions.docs(@normalize_schema)}

  ## Examples

      iex> t = Nx.tensor([[0, 0, 0], [3, 4, 5], [-2, 4, 3]])
      iex> Scholar.Preprocessing.Normalizer.fit_transform(t, axes: [1])
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, 0.0, 0.0],
          [0.42426407, 0.56568545, 0.70710677],
          [-0.37139067, 0.74278134, 0.55708605]
        ]
      >

      iex> t = Nx.tensor([[0, 0, 0], [3, 4, 5], [-2, 4, 3]])
      iex> Scholar.Preprocessing.Normalizer.fit_transform(t)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, 0.0, 0.0],
          [0.33752638, 0.45003518, 0.562544],
          [-0.22501759, 0.45003518, 0.33752638]
        ]
      >
  """
  deftransform fit_transform(tensor, opts \\ []) do
    normalize_n(tensor, NimbleOptions.validate!(opts, @normalize_schema))
  end

  defnp normalize_n(tensor, opts) do
    shape = Nx.shape(tensor)
    type = to_float_type(tensor)
    zeros = Nx.broadcast(Nx.tensor(0.0, type: type), shape)

    norm =
      case opts[:norm] do
        :euclidean ->
          Scholar.Metrics.Distance.euclidean(tensor, zeros, axes: opts[:axes])

        :manhattan ->
          Scholar.Metrics.Distance.manhattan(tensor, zeros, axes: opts[:axes])

        :chebyshev ->
          Scholar.Metrics.Distance.chebyshev(tensor, zeros, axes: opts[:axes])

        other ->
          raise ArgumentError,
                "expected :norm to be one of: :euclidean, :manhattan, and :chebyshev, got: #{inspect(other)}"
      end

    shape_to_broadcast = unsqueezed_reduced_shape(shape, opts[:axes])

    norm =
      Nx.select(norm == 0.0, Nx.tensor(1.0, type: type), norm) |> Nx.reshape(shape_to_broadcast)

    tensor / norm
  end

  deftransformp unsqueezed_reduced_shape(shape, axes) do
    if axes != nil do
      Enum.reduce(axes, shape, &put_elem(&2, &1, 1))
    else
      Tuple.duplicate(1, Nx.rank(shape))
    end
  end
end
