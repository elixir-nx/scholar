defmodule Scholar.Preprocessing.Binarizer do
  @moduledoc """
    Binarize data according to a threshold.
  """
  import Nx.Defn

  binarize_schema = [
    threshold: [
      type: :float,
      default: 0.0,
      doc: """
      Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.
      """
    ]
  ]

  @binarize_schema NimbleOptions.new!(binarize_schema)

  @doc """
  Values greater than the threshold map to 1, while values less than
    or equal to the threshold map to 0. With the default threshold of 0,
    only positive values map to 1.
  ## Options
  #{NimbleOptions.docs(@binarize_schema)}
  ## Examples
      iex> t = Nx.tensor([[0, 0, 0], [3, 4, 5], [-2, 4, 3]])
      iex> Scholar.Preprocessing.Binarizer.fit_transform(t, threshold: 3.0)
      #Nx.Tensor<
        u8[3][3]
        [
          [0, 0, 0],
          [0, 1, 1],
          [0, 1, 0]
        ]
      >
      iex> t = Nx.tensor([[0, 0, 0], [3, 4, 5], [-2, 4, 3]])
      iex> Scholar.Preprocessing.Binarizer.fit_transform(t,threshold: 0.4)
      #Nx.Tensor<
        u8[3][3]
        [
          [0, 0, 0],
          [1, 1, 1],
          [0, 1, 1]
        ]
      >
  """
  deftransform fit_transform(tensor, opts \\ []) do
    binarize_n(tensor, NimbleOptions.validate!(opts, @binarize_schema))
  end

  defnp binarize_n(tensor, opts) do
    threshold = opts[:threshold]
    tensor > threshold
  end
end
