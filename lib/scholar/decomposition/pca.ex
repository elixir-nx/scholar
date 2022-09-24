defmodule Scholar.Decomposition.PCA do
  @moduledoc """
  PCA decomposition algorithm.
  """
  import Nx.Defn

  @derive {Nx.Container,
           containers: [
             :components,
             :explained_variance,
             :explained_variance_ratio,
             :singular_values,
             :mean,
             :num_components,
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
      type: :any,
      default: :none,
      doc: ~S"""
      Number of components to keep.

      It may be a number or a tensor. When an integer number/tensor,
      it is the exact number of components. When a float number/tensor,
      it selects the number of components such that the amount of
      variance that needs to be explained is greater than the percentage
      specified by `:num_components`.

      If none is given, it defaults `:none`, and it picks the minimum
      between the number of samples and the number of features.
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
    opts = NimbleOptions.validate!(opts, @fit_opts_schema)

    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected input to have shape {n_samples, n_features}, got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    num_components =
      case opts[:num_components] do
        :none ->
          {num_samples, num_features} = Nx.shape(x)
          Kernel.min(num_features, num_samples)

        %Nx.Tensor{} = num ->
          num

        num when is_number(num) ->
          num

        num ->
          raise ArgumentError,
                ":num_components must be :none, a number, or a tensor, got: #{inspect(num)}"
      end

    fit_n(x, num_components)
  end

  defnp fit_n(x, num_components) do
    {num_samples, num_features} = Nx.shape(x)
    mean = Nx.mean(x, axes: [0])
    x = x - mean
    {decomposer, singular_values, components} = Nx.LinAlg.svd(x)
    explained_variance = singular_values * singular_values / (num_samples - 1)
    explained_variance_ratio = explained_variance / Nx.sum(explained_variance)

    type = Nx.type(num_components)
    clipped = Nx.clip(num_components, 0, min(num_samples, num_features))

    if Nx.rank(num_components) != 0 do
      raise ArgumentError,
            ":num_components must be a scalar tensor, got: #{inspect(num_components)}"
    end

    num_components =
      cond do
        Nx.Type.integer?(type) ->
          clipped

        Nx.Type.float?(type) ->
          Nx.cumulative_sum(explained_variance)
          |> then(&Nx.greater_equal(clipped, &1))
          |> Nx.as_type({:s, 8})
          |> Nx.argmax(tie_break: :low)
          # add one to change an index into the number of clusters
          |> Nx.add(1)

        true ->
          raise ArgumentError,
                ":num_components must be an integer or float tensor, got: #{inspect(num_components)}"
      end

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
end
