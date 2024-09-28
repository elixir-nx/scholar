defmodule Scholar.Preprocessing do
  @moduledoc """
  Set of functions for preprocessing data.
  """

  import Nx.Defn

  binarize_schema = [
    type: [
      type: {:custom, Scholar.Options, :type, []},
      default: :f32,
      doc: """
      Type of the resultant tensor.
      """
    ],
    threshold: [
      type: {:or, [:integer, :float]},
      default: 0,
      doc: """
      Feature values below or equal to this are replaced by 0, above it by 1.
      """
    ]
  ]

  @binarize_schema NimbleOptions.new!(binarize_schema)

  @doc """
  Standardizes the tensor by removing the mean and scaling to unit variance.

  It is a shortcut for `Scholar.Preprocessing.StandardScale.fit_transform/3`.
  See `Scholar.Preprocessing.StandardScale` for more information.

  ## Examples

      iex> Scholar.Preprocessing.standard_scale(Nx.tensor([1,2,3]))
      #Nx.Tensor<
        f32[3]
        [-1.2247447967529297, 0.0, 1.2247447967529297]
      >

  """
  defn standard_scale(tensor, opts \\ []) do
    Scholar.Preprocessing.StandardScaler.fit_transform(tensor, opts)
  end

  @doc """
  Scales a tensor by dividing each sample in batch by maximum absolute value in the batch.

  It is a shortcut for `Scholar.Preprocessing.MaxAbsScaler.fit_transform/2`.
  See `Scholar.Preprocessing.MaxAbsScaler` for more information.

  ## Examples

      iex> Scholar.Preprocessing.max_abs_scale(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f32[3]
        [0.3333333432674408, 0.6666666865348816, 1.0]
      >

      iex> Scholar.Preprocessing.max_abs_scale(Nx.tensor([[1, -1, 2], [3, 0, 0], [0, 1, -1], [2, 3, 1]]), axes: [0])
      #Nx.Tensor<
        f32[4][3]
        [
          [0.3333333432674408, -0.3333333432674408, 1.0],
          [1.0, 0.0, 0.0],
          [0.0, 0.3333333432674408, -0.5],
          [0.6666666865348816, 1.0, 0.5]
        ]
      >

      iex> Scholar.Preprocessing.max_abs_scale(42)
      #Nx.Tensor<
        f32
        1.0
      >
  """
  defn max_abs_scale(tensor, opts \\ []) do
    Scholar.Preprocessing.MaxAbsScaler.fit_transform(tensor, opts)
  end

  @doc """
  Scales a tensor by a given range.

  It is a shortcut for `Scholar.Preprocessing.MinMaxScaler.fit_transform/2`.
  See `Scholar.Preprocessing.MinMaxScaler` for more information.

  ## Examples

      iex> Scholar.Preprocessing.min_max_scale(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f32[3]
        [0.0, 0.5, 1.0]
      >

      iex> Scholar.Preprocessing.min_max_scale(42)
      #Nx.Tensor<
        f32
        0.0
      >
  """
  defn min_max_scale(tensor, opts \\ []) do
    Scholar.Preprocessing.MinMaxScaler.fit_transform(tensor, opts)
  end

  @doc """
  Converts a tensor into binary values based on the given threshold.

  ## Options

  #{NimbleOptions.docs(@binarize_schema)}

  ## Examples

      iex> Scholar.Preprocessing.binarize(Nx.tensor([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]))
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.0, 1.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0]
        ]
      >

      iex> Scholar.Preprocessing.binarize(Nx.tensor([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]), threshold: 1.3, type: {:u, 8})
      #Nx.Tensor<
        u8[3][3]
        [
          [0, 0, 1],
          [1, 0, 0],
          [0, 0, 0]
        ]
      >
  """
  deftransform binarize(tensor, opts \\ []) do
    binarize_n(tensor, NimbleOptions.validate!(opts, @binarize_schema))
  end

  defnp binarize_n(tensor, opts) do
    (tensor > opts[:threshold]) |> Nx.as_type(opts[:type])
  end

  @doc """
  It is a shortcut for `Scholar.Preprocessing.OrdinalEncoder.fit_transform/2`.
  See `Scholar.Preprocessing.OrdinalEncoder` for more information.

  ## Examples

      iex> Scholar.Preprocessing.ordinal_encode(Nx.tensor([3, 2, 4, 56, 2, 4, 2]))
      #Nx.Tensor<
        u64[7]
        [1, 0, 2, 3, 0, 2, 0]
      >
  """
  defn ordinal_encode(tensor) do
    Scholar.Preprocessing.OrdinalEncoder.fit_transform(tensor)
  end

  @doc """
  It is a shortcut for `Scholar.Preprocessing.OneHotEncoder.fit_transform/2`.
  See `Scholar.Preprocessing.OneHotEncoder` for more information.

  ## Examples

      iex> Scholar.Preprocessing.one_hot_encode(Nx.tensor([2, 0, 3, 2, 1, 1, 0]), num_categories: 4)
      #Nx.Tensor<
        u8[7][4]
        [
          [0, 0, 1, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 1, 0],
          [0, 1, 0, 0],
          [0, 1, 0, 0],
          [1, 0, 0, 0]
        ]
      >
  """
  defn one_hot_encode(tensor, opts) do
    Scholar.Preprocessing.OneHotEncoder.fit_transform(tensor, opts)
  end

  @doc """
  Normalize samples individually to unit norm.

  The zero-tensors cannot be normalized and they stay the same
  after normalization.

  It is a shortcut for `Scholar.Preprocessing.Normalizer.fit_transform/2`.
  See `Scholar.Preprocessing.Normalizer` for more information.

  ## Examples

      iex> Scholar.Preprocessing.normalize(Nx.tensor([[0, 0, 0], [3, 4, 5], [-2, 4, 3]]), axes: [1])
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, 0.0, 0.0],
          [0.4242640733718872, 0.5656854510307312, 0.7071067690849304],
          [-0.3713906705379486, 0.7427813410758972, 0.5570860505104065]
        ]
      >

      iex> Scholar.Preprocessing.normalize(Nx.tensor([[0, 0, 0], [3, 4, 5], [-2, 4, 3]]))
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, 0.0, 0.0],
          [0.3375263810157776, 0.4500351846218109, 0.5625439882278442],
          [-0.22501759231090546, 0.4500351846218109, 0.3375263810157776]
        ]
      >
  """
  deftransform normalize(tensor, opts \\ []) do
    Scholar.Preprocessing.Normalizer.fit_transform(tensor, opts)
  end
end
