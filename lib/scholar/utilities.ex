defmodule Scholar.Utilities do
  @moduledoc """
  Utility functions for general purposes.
  """

  import Nx.Defn

  train_test_split_schema = [
    train_size: [
      required: true,
      type: :float,
      doc: """
      Percentual size for the training set.
      """
    ]
  ]

  @train_test_split_schema NimbleOptions.new!(train_test_split_schema)

  @doc ~S"""
  Split tensors into random train and test subsets.

  ## Options

  #{NimbleOptions.docs(@train_test_split_schema)}

  ## Example

      Split into 80% for training and 20% for testing.

      iex> {_train, _test} = Scholar.Utilities.train_test_split(Nx.tensor([[3, 6, 5], [26, 75, 3], [23, 4, 1]]), train_size: 0.8)
      {#Nx.Tensor<
        s64[2][3]
        [
          [3, 6, 5],
          [26, 75, 3]
        ]
      >,
      #Nx.Tensor<
        s64[1][3]
        [
          [23, 4, 1]
        ]
      >}
  """
  deftransform train_test_split(%Nx.Tensor{} = features_tensor, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @train_test_split_schema)

    features = Nx.to_list(features_tensor)
    num_samples = Enum.count(features)
    split_size = floor(opts[:train_size] * num_samples)
    {train, test} = Enum.split(features, split_size)
    {Nx.tensor(train), Nx.tensor(test)}
  end
end
