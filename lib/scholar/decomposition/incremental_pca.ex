defmodule Scholar.Decomposition.IncrementalPCA do
  @moduledoc """
  Incremental Principal Component Analysis.

  Performs linear dimensionality reduction by processing the input data
  in batches and incrementally updating the principal components.
  This iterative approach is particularly suitable for datasets too large to fit in memory,
  as its memory complexity is independent of the number of data samples.

  References:

  * [1] [Incremental Learning for Robust Visual Tracking](https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf)
  """
  import Nx.Defn
  require Nx

  @derive {Nx.Container,
           keep: [:whiten?],
           containers: [
             :components,
             :singular_values,
             :num_samples_seen,
             :mean,
             :variance,
             :explained_variance,
             :explained_variance_ratio
           ]}
  defstruct [
    :components,
    :singular_values,
    :num_samples_seen,
    :mean,
    :variance,
    :explained_variance,
    :explained_variance_ratio,
    :whiten?
  ]

  opts = [
    num_components: [
      required: true,
      type: :pos_integer,
      doc: "The number of principal components."
    ],
    whiten?: [
      type: :boolean,
      default: false,
      doc: """
      When true the `components` are divided by `num_samples` times `components` to ensure uncorrelated outputs with unit component-wise variances.

      Whitening will remove some information from the transformed signal (the relative variance scales of the components)
      but can sometimes improve the predictive accuracy of the downstream estimators by making data respect some hard-wired assumptions.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits an Incremental PCA model on a stream of batches.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return values

  The function returns a struct with the following parameters:

    * `:num_components` - The number of principal components.

    * `:components` - Principal axes in feature space, representing the directions of maximum variance in the data.
      Equivalently, the right singular vectors of the centered input data, parallel to its eigenvectors.
      The components are sorted by decreasing `:explained_variance`.

    * `:singular_values` - The singular values corresponding to each of the selected components.
      The singular values are equal to the 2-norms of the `:num_components` variables in the lower-dimensional space.

    * `:num_samples_seen` - The number of data samples processed.

    * `:mean` - Per-feature empirical mean.

    * `:variance` - Per-feature empirical variance.

    * `:explained_variance` - Variance explained by each of the selected components.

    * `:explained_variance_ratio` - Percentage of variance explained by each of the selected components.

    * `:whiten?` - Whether to apply whitening.

  ## Examples

      iex> {x, _} = Scidata.Iris.download()
      iex> batches = x |> Nx.tensor() |> Nx.to_batched(10)
      iex> ipca = Scholar.Decomposition.IncrementalPCA.fit(batches, num_components: 2)
      iex> ipca.components
      Nx.tensor(
        [
          [-0.33354005217552185, 0.1048964187502861, -0.8618107080105579, -0.3674643635749817],
          [-0.5862125754356384, -0.7916879057884216, 0.15874788165092468, -0.06621300429105759]
        ]
      )
      iex> ipca.singular_values
      Nx.tensor([77.05782028025969, 10.137848854064941])
  """
  def fit(batches = %Stream{}, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    Enum.reduce(
      batches,
      nil,
      fn batch, model -> fit_batch(model, batch, opts) end
    )
  end

  defp fit_batch(nil, batch, opts), do: fit_head_n(batch, opts)
  defp fit_batch(%__MODULE__{} = model, batch, _opts), do: partial_fit(model, batch)

  deftransformp fit_head(x, opts) do
    {num_samples, num_features} = Nx.shape(x)
    num_components = opts[:num_componenets]

    cond do
      num_components > num_samples ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              batch_size = #{num_samples}, got #{num_components}
              """

      num_components > num_features ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_features = #{num_features}, got #{num_components}
              """
    end

    fit_head_n(x, opts)
  end

  defnp fit_head_n(x, opts) do
    # This is similar to Scholar.Decomposition.PCA.fit_n
    num_components = opts[:num_components]
    num_samples = Nx.u64(Nx.axis_size(x, 0))
    mean = Nx.mean(x, axes: [0])
    variance = Nx.variance(x, axes: [0])
    x_centered = x - mean
    {u, s, vt} = Nx.LinAlg.svd(x_centered, full_matrices?: false)
    {_, vt} = Scholar.Decomposition.Utils.flip_svd(u, vt)
    components = vt[[0..(num_components - 1)]]
    singular_values = s[[0..(num_components - 1)]]
    explained_variance = s * s / (num_samples - 1)

    explained_variance_ratio =
      (explained_variance / Nx.sum(explained_variance))[[0..(num_components - 1)]]

    %__MODULE__{
      components: components,
      singular_values: singular_values,
      num_samples_seen: num_samples,
      mean: mean,
      variance: variance,
      explained_variance: explained_variance[[0..(num_components - 1)]],
      explained_variance_ratio: explained_variance_ratio,
      whiten?: opts[:whiten?]
    }
  end

  @doc false
  deftransform partial_fit(model, x) do
    {num_components, num_features_seen} = Nx.shape(model.components)
    {num_samples, num_features} = Nx.shape(x)

    cond do
      num_features_seen != num_features ->
        raise ArgumentError,
              """
              each batch must have the same number of features, \
              got #{num_features_seen} and #{num_features}
              """

      num_components > num_samples ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              batch_size = #{num_samples}, got #{num_components}
              """

      true ->
        nil
    end

    partial_fit_n(model, x)
  end

  defnp partial_fit_n(model, x) do
    components = model.components
    num_components = Nx.axis_size(components, 0)
    singular_values = model.singular_values
    num_samples_seen = model.num_samples_seen
    mean = model.mean
    variance = model.variance
    {num_samples, _} = Nx.shape(x)

    {x_mean, x_centered, new_num_samples_seen, new_mean, new_variance} =
      incremental_mean_and_variance(x, num_samples_seen, mean, variance)

    mean_correction =
      Nx.sqrt(num_samples_seen / new_num_samples_seen) * num_samples * (mean - x_mean)

    mean_correction = Nx.new_axis(mean_correction, 0)

    matrix =
      Nx.concatenate(
        [
          Nx.new_axis(singular_values, 1) * components,
          x_centered,
          mean_correction
        ],
        axis: 0
      )

    {u, s, vt} = Nx.LinAlg.svd(matrix, full_matrices?: false)
    {_, vt} = Scholar.Decomposition.Utils.flip_svd(u, vt)
    new_components = vt[[0..(num_components - 1)]]
    new_singular_values = s[[0..(num_components - 1)]]
    new_explained_variance = singular_values * singular_values / (new_num_samples_seen - 1)

    new_explained_variance_ratio =
      singular_values * singular_values / Nx.sum(new_variance * new_num_samples_seen)

    %__MODULE__{
      components: new_components,
      singular_values: new_singular_values,
      num_samples_seen: new_num_samples_seen,
      mean: new_mean,
      variance: new_variance,
      explained_variance: new_explained_variance,
      explained_variance_ratio: new_explained_variance_ratio,
      whiten?: model.whiten?
    }
  end

  defnp incremental_mean_and_variance(x, num_samples_seen, mean, variance) do
    num_samples = Nx.axis_size(x, 0)
    new_num_samples_seen = num_samples_seen + num_samples
    sum = num_samples_seen * mean
    x_sum = Nx.sum(x, axes: [0])
    new_mean = (sum + x_sum) / new_num_samples_seen
    x_mean = x_sum / num_samples
    x_centered = x - x_mean
    correction = Nx.sum(x_centered, axes: [0])

    x_unnormalized_variance =
      Nx.sum(x_centered * x_centered, axes: [0]) - correction * correction / num_samples

    unnormalized_variance = num_samples_seen * variance
    last_over_new_count = num_samples_seen / num_samples

    new_unnormalized_variance =
      unnormalized_variance +
        x_unnormalized_variance +
        last_over_new_count / new_num_samples_seen *
          (sum / last_over_new_count - x_sum) ** 2

    new_variance = new_unnormalized_variance / new_num_samples_seen
    {x_mean, x_centered, new_num_samples_seen, new_mean, new_variance}
  end

  @doc """
  Applies dimensionality reduction to the data `x` using Incremental PCA `model`.

  ## Examples

      iex> {x, _} = Scidata.Iris.download()
      iex> batches = x |> Nx.tensor() |> Nx.to_batched(10)
      iex> ipca = Scholar.Decomposition.IncrementalPCA.fit(batches, num_components: 2)
      iex> x = Nx.tensor([[5.2, 2.6, 2.475, 0.7], [6.1, 3.2, 3.95, 1.3], [7.0, 3.8, 5.425, 1.9]])
      iex> Scholar.Decomposition.IncrementalPCA.transform(ipca, x)
      Nx.tensor(
        [
          [1.4564743041992188, 0.5657951235771179],
          [-0.27242332696914673, -0.24238374829292297],
          [-2.0013210773468018, -1.0505625009536743]
        ]
      )
  """
  deftransform transform(model, x) do
    transform_n(model, x)
  end

  defnp transform_n(
          %__MODULE__{
            components: components,
            explained_variance: explained_variance,
            mean: mean,
            whiten?: whiten?
          } = _model,
          x
        ) do
    # This is literally the same as Scholar.Decomposition.PCA.transform_n!
    z = Nx.dot(x - mean, [1], components, [1])

    if whiten? do
      z / Nx.sqrt(explained_variance)
    else
      z
    end
  end
end
