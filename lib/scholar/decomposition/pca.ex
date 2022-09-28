defmodule Scholar.Decomposition.PCA do
  @moduledoc """
  PCA decomposition algorithm.
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
             :num_samples,
             :decomposer
           ]}
  defstruct [
    :components,
    :explained_variance,
    :explained_variance_ratio,
    :singular_values,
    :mean,
    :num_components,
    :num_features,
    :num_samples,
    :decomposer
  ]

  fit_opts_schema = [
    num_components: [
      type:
        {:or, [{:custom, Scholar.Options, :positive_number, []}, {:in, [:none, :pos_integer]}]},
      default: :none,
      doc: ~S"""
      Number of components to keep. if `:num_components` is not set all components are kept:

      $num\\_components = min(num\\_samples, num\\_features)$

      If $0 < num\\_components <= 1$, select the number of components such
      that the amount of variance that needs to be explained is greater than the percentage specified by `:num_components`.
      """
    ]
  ]

  transform_opts_schema = [
    num_components: [
      required: true,
      type: :pos_integer,
      doc: """
      Number of components to keep. It should equal to the value calculated in the `fit/2` method.
      """
    ],
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

  @fit_opts_schema NimbleOptions.new!(fit_opts_schema)
  @transform_opts_schema NimbleOptions.new!(transform_opts_schema)

  @doc """
  Fits a PCA for sample inputs `x`.

  ## Options

  #{NimbleOptions.docs(@fit_opts_schema)}

  ## Returns

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

    * `:num_components` - The estimated number of components. When `:num_components` is set to a number between 0 and 1
      this number is estimated from input data. Otherwise it equals the parameter `:num_components`, or the lesser
      value of `:num_features` and `:num_samples` if `num_components` is `nil`.

    * `:num_features` - Number of features in the training data.

    * `:num_samples` - Number of samples in the training data.

    * `:decomposer` - Matrix used in actual decomposition. It is the unitary matrix from SVD factorization.
  """
  deftransform fit(x, opts \\ []) do
    fit_n(x, NimbleOptions.validate!(opts, @fit_opts_schema))
  end

  defnp fit_n(x, opts \\ []) do
    if Nx.rank(x) != 2 do
      raise ArgumentError, "expected x to have rank equal to: 2, got: #{inspect(Nx.rank(x))}"
    end

    {num_samples, num_features} = Nx.shape(x)
    num_components = opts[:num_components]

    mean = Nx.mean(x, axes: [0])
    x = x - mean
    {decomposer, singular_values, components} = Nx.LinAlg.svd(x)
    explained_variance = singular_values * singular_values / (num_samples - 1)
    explained_variance_ratio = explained_variance / Nx.sum(explained_variance)

    num_components =
      calculate_num_components(
        num_components,
        num_features,
        num_samples,
        explained_variance
      )

    %__MODULE__{
      components: components,
      explained_variance: explained_variance,
      explained_variance_ratio: explained_variance_ratio,
      singular_values: singular_values,
      mean: mean,
      num_components: num_components,
      num_features: num_features,
      num_samples: num_samples,
      decomposer: decomposer
    }
  end

  @doc """
  For a fitted model performs a decomposition.

  ## Options

  #{NimbleOptions.docs(@transform_opts_schema)}

  ## Returns

  The function returns a decomposed data.
  """
  deftransform transform(model, opts \\ []) do
    transform_n(model, NimbleOptions.validate!(opts, @transform_opts_schema))
  end

  defnp transform_n(
          %__MODULE__{
            singular_values: singular_values,
            num_samples: num_samples,
            decomposer: decomposer
          } = _model,
          opts \\ []
        ) do
    opts = keyword!(opts, [:num_components, whiten: false])
    num_components = opts[:num_components]
    whiten? = opts[:whiten]

    if not whiten? do
      decomposer[[0..-1//1, 0..(num_components - 1)]] * singular_values
    else
      decomposer[[0..-1//1, 0..(num_components - 1)]] * Nx.sqrt(num_samples - 1)
    end
  end

  deftransformp calculate_num_components(
                  num_components,
                  num_features,
                  num_samples,
                  explained_variance
                ) do
    cond do
      num_components == :none ->
        Nx.min(num_features, num_samples)

      num_components > 0 and num_components <= 1 ->
        Nx.cumulative_sum(explained_variance)
        |> then(&Nx.greater_equal(num_components, &1))
        |> Nx.as_type({:s, 8})
        |> Nx.argmax(tie_break: :low)
        # add one to change an index into the number of clusters
        |> Nx.add(1)

      num_components > 0 and num_components <= min(num_features, num_samples) and
          is_integer(num_components) ->
        num_components

      is_integer(num_components) ->
        raise ArgumentError,
              "expected :num_components to be integer in range 1 to #{inspect(min(num_samples, num_features))}, got: #{inspect(num_components)}"

      is_float(num_components) ->
        raise ArgumentError,
              "expected :num_components to be float in range 0 to 1, got: #{inspect(num_components)}"

      true ->
        raise ArgumentError, "unexpected type of :num_components, got: #{inspect(num_components)}"
    end
  end
end
