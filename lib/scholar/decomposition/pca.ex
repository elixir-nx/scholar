defmodule Scholar.Decomposition.PCA do
  @moduledoc """
  Principal Component Analysis (PCA).

  The main concept of PCA is to find components (i.e. columns of a matrix) which explain the most variance
  of data set [1]. The sample data is decomposed using linear combination of
  vectors that lie on the directions of those components.

  The time complexity is $O(NP^2 + P^3)$ where $N$ is the number of samples and $P$ is the number of features.
  Space complexity is $O(P * (P+N))$.
  Reference:

  * [1] - [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
  """
  import Nx.Defn

  @derive {Nx.Container,
           keep: [:num_components],
           containers: [
             :components,
             :explained_variance,
             :explained_variance_ratio,
             :singular_values,
             :mean,
             :num_features,
             :num_samples
           ]}
  defstruct [
    :components,
    :explained_variance,
    :explained_variance_ratio,
    :singular_values,
    :mean,
    :num_components,
    :num_features,
    :num_samples
  ]

  fit_opts_schema = [
    num_components: [
      type: {:or, [:pos_integer, {:in, [nil]}]},
      default: nil,
      doc: ~S"""
      Number of components to keep. If `:num_components` is not set, all components are kept
      which is the minimum value from number of features and number of samples.
      """
    ]
  ]

  transform_opts_schema = [
    whiten: [
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

  fit_transform_opts_schema = fit_opts_schema ++ transform_opts_schema

  @fit_opts_schema NimbleOptions.new!(fit_opts_schema)
  @transform_opts_schema NimbleOptions.new!(transform_opts_schema)
  @fit_transform_opts_schema NimbleOptions.new!(fit_transform_opts_schema)

  @doc """
  Fits a PCA for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@fit_opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:components` - Principal axes in feature space, representing the directions of maximum variance in the data.
      Equivalently, the right singular vectors of the centered input data, parallel to its eigenvectors.
      The components are sorted by `:explained_variance`.

    * `:explained_variance` - The amount of variance explained by each of the selected components.
      The variance estimation uses `:num_samples - 1` degrees of freedom.
      Equal to `:num_components` largest eigenvalues of the covariance matrix of `x`.

    * `:explained_variance_ratio` - Percentage of variance explained by each of the selected components.
      If `:num_components` is not set then all components are stored and the sum of the ratios is equal to 1.0.

    * `:singular_values` - The singular values corresponding to each of the selected components.
      The singular values are equal to the 2-norms of the `:num_components` variables in the lower-dimensional space.

    * `:mean` - Per-feature empirical mean, estimated from the training set.

    * `:num_components` - It equals the parameter `:num_components`, or the lesser
      value of `:num_features` and `:num_samples` if the parameter `:num_components` is `nil`.

    * `:num_features` - Number of features in the training data.

    * `:num_samples` - Number of samples in the training data.

  ## Examples
      iex> x = Nx.tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
      iex> Scholar.Decomposition.PCA.fit(x)
      %Scholar.Decomposition.PCA{
        components: Nx.tensor(
          [
            [-0.8387274146080017, -0.544551432132721],
            [0.544551432132721, -0.8387274146080017]
          ]
        ),
        explained_variance: Nx.tensor(
          [7.939539909362793, 0.060457102954387665]
        ),
        explained_variance_ratio: Nx.tensor(
          [0.9924428462982178, 0.007557140663266182]
        ),
        singular_values: Nx.tensor(
          [6.3006110191345215, 0.5498049855232239]
        ),
        mean: Nx.tensor(
          [0.0, 0.0]
        ),
        num_components: 2,
        num_features: Nx.tensor(
          2
        ),
        num_samples: Nx.tensor(
          6
        )
      }
  """
  deftransform fit(x, opts \\ []) do
    fit_n(x, NimbleOptions.validate!(opts, @fit_opts_schema))
  end

  # TODO Add support for :num_components as a float when dynamic shapes will be implemented
  defnp fit_n(x, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError, "expected x to have rank equal to: 2, got: #{inspect(Nx.rank(x))}"
    end

    {num_samples, num_features} = Nx.shape(x)
    num_components = opts[:num_components]

    mean = Nx.mean(x, axes: [0])
    x = x - mean
    {decomposer, singular_values, components} = Nx.LinAlg.svd(x, full_matrices?: false)

    num_components =
      calculate_num_components(
        num_components,
        num_features,
        num_samples
      )

    {_, components} = flip_svd(decomposer, components)
    components = components[[0..(num_components - 1), ..]]

    explained_variance = singular_values * singular_values / (num_samples - 1)

    explained_variance_ratio =
      (explained_variance / Nx.sum(explained_variance))[[0..(num_components - 1)]]

    %__MODULE__{
      components: components,
      explained_variance: explained_variance[[0..(num_components - 1)]],
      explained_variance_ratio: explained_variance_ratio,
      singular_values: singular_values[[0..(num_components - 1)]],
      mean: mean,
      num_components: num_components,
      num_features: num_features,
      num_samples: num_samples
    }
  end

  @doc """
  For a fitted `model` performs a decomposition.

  ## Options

  #{NimbleOptions.docs(@transform_opts_schema)}

  ## Return Values

  The function returns a tensor with decomposed data.

  ## Examples
      iex> x = Nx.tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
      iex> model = Scholar.Decomposition.PCA.fit(x)
      iex> Scholar.Decomposition.PCA.transform(model, x)
      Nx.tensor(
        [
          [1.3832788467407227, 0.29417598247528076],
          [2.222006320953369, -0.2503754496574402],
          [3.605285167694092, 0.043800532817840576],
          [-1.3832788467407227, -0.29417598247528076],
          [-2.222006320953369, 0.2503754496574402],
          [-3.605285167694092, -0.043800532817840576]
        ]
      )
  """
  deftransform transform(model, x, opts \\ []) do
    transform_n(model, x, NimbleOptions.validate!(opts, @transform_opts_schema))
  end

  defnp transform_n(
          %__MODULE__{
            components: components,
            explained_variance: explained_variance,
            mean: mean
          } = _model,
          x,
          opts
        ) do
    whiten? = opts[:whiten]

    x = x - mean

    x_transformed = Nx.dot(x, [1], components, [1])

    if whiten? do
      x_transformed / Nx.sqrt(explained_variance)
    else
      x_transformed
    end
  end

  @doc """
  Fit the model with `x` and apply the dimensionality reduction on `x`.

  This function is analogous to calling `fit/2` and then
  `transform/3`, but it is calculated more efficiently.

  ## Options

  #{NimbleOptions.docs(@transform_opts_schema)}

  ## Return Values

  The function returns a tensor with decomposed data.

  ## Examples

      iex> x = Nx.tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
      iex> Scholar.Decomposition.PCA.fit_transform(x)
      Nx.tensor(
        [
          [1.3819527626037598, 0.29363134503364563],
          [2.2231407165527344, -0.25125157833099365],
          [3.605093240737915, 0.04237978905439377],
          [-1.3819527626037598, -0.29363134503364563],
          [-2.2231407165527344, 0.25125157833099365],
          [-3.605093240737915, -0.04237978905439377]
        ]
      )
  """

  deftransform fit_transform(x, opts \\ []) do
    fit_transform_n(x, NimbleOptions.validate!(opts, @fit_transform_opts_schema))
  end

  defnp fit_transform_n(x, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError, "expected x to have rank equal to: 2, got: #{inspect(Nx.rank(x))}"
    end

    {num_samples, num_features} = Nx.shape(x)
    num_components = opts[:num_components]
    x = x - Nx.mean(x, axes: [0])
    {decomposer, singular_values, components} = Nx.LinAlg.svd(x, full_matrices?: false)

    num_components =
      calculate_num_components(
        num_components,
        num_features,
        num_samples
      )

    {decomposer, _components} = flip_svd(decomposer, components)
    decomposer = decomposer[[.., 0..(num_components - 1)]]

    if opts[:whiten] do
      decomposer * Nx.sqrt(num_samples - 1)
    else
      decomposer * singular_values[[0..(num_components - 1)]]
    end
  end

  defnp flip_svd(u, v) do
    # columns of u, rows of v
    max_abs_cols_idx = u |> Nx.abs() |> Nx.argmax(axis: 0, keep_axis: true)
    signs = u |> Nx.take_along_axis(max_abs_cols_idx, axis: 0) |> Nx.sign() |> Nx.squeeze()
    u = u * signs
    v = v * Nx.new_axis(signs, -1)
    {u, v}
  end

  deftransformp calculate_num_components(
                  num_components,
                  num_features,
                  num_samples
                ) do
    default_num_components = min(num_features, num_samples)

    cond do
      num_components == nil ->
        default_num_components

      num_components > 0 and num_components <= min(num_features, num_samples) and
          is_integer(num_components) ->
        num_components

      is_integer(num_components) ->
        raise ArgumentError,
              "expected :num_components to be integer in range 1 to #{inspect(min(num_samples, num_features))}, got: #{inspect(num_components)}"

      true ->
        raise ArgumentError, "unexpected type of :num_components, got: #{inspect(num_components)}"
    end
  end
end
