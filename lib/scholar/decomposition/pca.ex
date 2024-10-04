defmodule Scholar.Decomposition.PCA do
  @moduledoc """
  Principal Component Analysis (PCA).

  PCA is a method for reducing the dimensionality of the data by transforming the original features
  into a new set of uncorrelated features called principal components, which capture the maximum
  variance in the data.
  It can be trained on the entirety of the data at once using `fit/2` or
  incrementally for datasets that are too large to fit in the memory using `incremental_fit/2`.

  The time complexity is $O(NP^2 + P^3)$ where $N$ is the number of samples and $P$ is the number of features.
  Space complexity is $O(P * (P+N))$.

  References:

  * [1] Dimensionality Reduction with Principal Component Analysis. [Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf), Chapter 10
  * [2] [Incremental Learning for Robust Visual Tracking](https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf)
  """
  import Nx.Defn

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
      doc: "The number of principal components to keep."
    ],
    whiten?: [
      type: :boolean,
      default: false,
      doc: """
      When true the result is multiplied by the square root of `:num_samples` and then
      divided by the `:singular_values` to ensure uncorrelated outputs with unit component-wise variances.

      Whitening will remove some information from the transformed signal (the relative variance scales of the components)
      but can sometime improve the predictive accuracy of the downstream estimators by making their data respect some hard-wired assumptions.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a PCA for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:components` - Principal axes in feature space, representing the directions of maximum variance in the data.
      Equivalently, the right singular vectors of the centered input data, parallel to its eigenvectors.
      The components are sorted by decreasing `:explained_variance`.

    * `:singular_values` - The singular values corresponding to each of the selected components.
      The singular values are equal to the 2-norms of the `:num_components` variables in the lower-dimensional space.

    * `:num_samples_seen` - Number of samples in the training data.

    * `:mean` - Per-feature empirical mean, estimated from the training set.

    * `:variance` - Per-feature empirical variance.

    * `:explained_variance` - The amount of variance explained by each of the selected components.
      The variance estimation uses `:num_samples - 1` degrees of freedom.
      Equal to `:num_components` largest eigenvalues of the covariance matrix of `x`.

    * `:explained_variance_ratio` - Percentage of variance explained by each of the selected components.

    * `:whiten?` - Whether to apply whitening.

  ## Examples

      iex> x = Scidata.Iris.download() |> elem(0) |> Nx.tensor()
      iex> pca = Scholar.Decomposition.PCA.fit(x, num_components: 2)
      iex> pca.components
      Nx.tensor(
        [
          [0.36182016134262085, -0.08202514797449112, 0.8565111756324768, 0.3588128685951233],
          [0.6585038900375366, 0.7275884747505188, -0.17632202804088593, -0.07679986208677292]
        ]
      )
      iex> pca.singular_values
      Nx.tensor([25.089859008789062, 6.007821559906006])
  """
  deftransform fit(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}\
            """
    end

    {num_samples, num_features} = Nx.shape(x)
    num_components = opts[:num_components]

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

      true ->
        nil
    end

    fit_n(x, opts)
  end

  defnp fit_n(x, opts) do
    num_samples = Nx.axis_size(x, 0) |> Nx.u64()
    num_components = opts[:num_components]

    mean = Nx.mean(x, axes: [0])
    x_centered = x - mean
    variance = Nx.sum(x_centered * x_centered / (num_samples - 1), axes: [0])
    {u, s, vt} = Nx.LinAlg.svd(x_centered, full_matrices?: false)
    {_, vt} = Scholar.Decomposition.Utils.flip_svd(u, vt)
    components = vt[0..(num_components - 1)]
    explained_variance = s * s / (num_samples - 1)

    explained_variance_ratio =
      (explained_variance / Nx.sum(explained_variance))[0..(num_components - 1)]

    %__MODULE__{
      components: components,
      singular_values: s[0..(num_components - 1)],
      num_samples_seen: num_samples,
      mean: mean,
      variance: variance,
      explained_variance: explained_variance[0..(num_components - 1)],
      explained_variance_ratio: explained_variance_ratio,
      whiten?: opts[:whiten?]
    }
  end

  @doc """
  Fits a PCA model on a stream of batches.

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
      iex> pca = Scholar.Decomposition.PCA.incremental_fit(batches, num_components: 2)
      iex> pca.components
      Nx.tensor(
        [
          [-0.33354005217552185, 0.1048964187502861, -0.8618107080105579, -0.3674643635749817],
          [-0.5862125754356384, -0.7916879057884216, 0.15874788165092468, -0.06621300429105759]
        ]
      )
      iex> pca.singular_values
      Nx.tensor([77.05782028025969, 10.137848854064941])
  """
  deftransform incremental_fit(batches, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    Enum.reduce(
      batches,
      nil,
      fn batch, model -> fit_batch(model, batch, opts) end
    )
  end

  defp fit_batch(nil, batch, opts), do: fit(batch, opts)
  defp fit_batch(%__MODULE__{} = model, batch, _opts), do: partial_fit(model, batch)

  @doc """
  Updates the parameters of a PCA model on samples `x`.

  ## Examples

    iex> {x, _} = Scidata.Iris.download()
    iex> {first_batch, second_batch} = x |> Nx.tensor() |> Nx.split(75)
    iex> pca = Scholar.Decomposition.PCA.fit(first_batch, num_components: 2)
    iex> pca = Scholar.Decomposition.PCA.partial_fit(pca, second_batch)
    iex> pca.components
    Nx.tensor(
      [
        [-0.3229745328426361, 0.09587063640356064, -0.8628664612770081, -0.37677285075187683],
        [-0.6786625981330872, -0.7167785167694092, 0.14237160980701447, 0.07332050055265427]
      ]
    )
    iex> pca.singular_values
    Nx.tensor([166.141845703125, 6.078948020935059])
  """
  deftransform partial_fit(model, x) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}\
            """
    end

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
              num_samples = #{num_samples}, got #{num_components}
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

    {x_mean, x_centered, updated_num_samples_seen, updated_mean, updated_variance} =
      incremental_mean_and_variance(x, num_samples_seen, mean, variance)

    mean_correction =
      Nx.sqrt(num_samples_seen / updated_num_samples_seen) * num_samples * (mean - x_mean)

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
    updated_components = vt[0..(num_components - 1)]
    updated_singular_values = s[0..(num_components - 1)]

    updated_explained_variance =
      singular_values * singular_values / (updated_num_samples_seen - 1)

    updated_explained_variance_ratio =
      singular_values * singular_values / Nx.sum(updated_variance * updated_num_samples_seen)

    %__MODULE__{
      components: updated_components,
      singular_values: updated_singular_values,
      num_samples_seen: updated_num_samples_seen,
      mean: updated_mean,
      variance: updated_variance,
      explained_variance: updated_explained_variance,
      explained_variance_ratio: updated_explained_variance_ratio,
      whiten?: model.whiten?
    }
  end

  defnp incremental_mean_and_variance(x, num_samples_seen, mean, variance) do
    new_num_samples = Nx.axis_size(x, 0)
    updated_num_samples_seen = num_samples_seen + new_num_samples
    sum = num_samples_seen * mean
    new_sum = Nx.sum(x, axes: [0])
    updated_mean = (sum + new_sum) / updated_num_samples_seen
    new_mean = new_sum / new_num_samples
    x_centered = x - new_mean
    correction = Nx.sum(x_centered, axes: [0])

    new_unnormalized_variance =
      Nx.sum(x_centered * x_centered, axes: [0]) - correction * correction / new_num_samples

    unnormalized_variance = num_samples_seen * variance
    seen_over_new = num_samples_seen / new_num_samples

    updated_unnormalized_variance =
      unnormalized_variance +
        new_unnormalized_variance +
        seen_over_new / updated_num_samples_seen *
          (sum / seen_over_new - new_sum) ** 2

    updated_variance = updated_unnormalized_variance / updated_num_samples_seen
    {new_mean, x_centered, updated_num_samples_seen, updated_mean, updated_variance}
  end

  @doc """
  For a fitted `model` performs a decomposition of samples `x`.

  ## Return Values

  The function returns a tensor with decomposed data.

  ## Examples
      iex> x_fit = Scidata.Iris.download() |> elem(0) |> Nx.tensor()
      iex> pca = Scholar.Decomposition.PCA.fit(x_fit, num_components: 2)
      iex> x_transform = Nx.tensor([[5.2, 2.6, 2.475, 0.7], [6.1, 3.2, 3.95, 1.3], [7.0, 3.8, 5.425, 1.9]])
      iex> Scholar.Decomposition.PCA.transform(pca, x_transform)
      Nx.tensor(
        [
          [-1.4739344120025635, -0.48932668566703796],
          [0.28113049268722534, 0.2337251454591751],
          [2.0361955165863037, 0.9567767977714539]
        ]
      )
  """
  deftransform transform(model, x) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            """
            expected input tensor to have shape {num_samples, num_features}, \
            got tensor with shape: #{inspect(Nx.shape(x))}\
            """
    end

    num_features_seen = Nx.axis_size(model.components, 1)
    num_features = Nx.axis_size(x, 1)

    if num_features_seen != num_features do
      raise ArgumentError,
            """
            expected input tensor to have the same number of features \
            as tensor used to fit the model, \
            got #{inspect(num_features)} \
            and #{inspect(num_features_seen)}
            """
    end

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
    x_centered = x - mean

    z = Nx.dot(x_centered, [1], components, [1])

    if whiten? do
      z / Nx.sqrt(explained_variance)
    else
      z
    end
  end

  @doc """
  Fit the model with `x` and apply the dimensionality reduction on `x`.

  This function is equivalent to calling `fit/2` and then
  `transform/3`, but the result is computed more efficiently.

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a tensor with decomposed data.

  ## Examples

      iex> x = Scidata.Iris.download() |> elem(0) |> Enum.take(6) |> Nx.tensor()
      iex> Scholar.Decomposition.PCA.fit_transform(x, num_components: 2)
      Nx.tensor(
        [
          [0.16441848874092102, 0.028548287227749825],
          [-0.32804328203201294, 0.20709986984729767],
          [-0.3284338414669037, -0.08318747580051422],
          [-0.42237386107444763, -0.0735677033662796],
          [0.17480169236660004, -0.11189625412225723],
          [0.7396301627159119, 0.03300142288208008
          ]
        ]
      )
  """
  deftransform fit_transform(x, opts) do
    fit_transform_n(x, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_transform_n(x, opts) do
    num_components = opts[:num_components]
    mean = Nx.mean(x, axes: [0])
    x_centered = x - mean
    {u, s, vt} = Nx.LinAlg.svd(x_centered, full_matrices?: false)
    {u, _} = Scholar.Decomposition.Utils.flip_svd(u, vt)
    u = u[[.., 0..(num_components - 1)]]

    if opts[:whiten?] do
      u * Nx.sqrt(Nx.axis_size(x, 0) - 1)
    else
      u * s[0..(num_components - 1)]
    end
  end
end
