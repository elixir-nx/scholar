defmodule Scholar.Decomposition.IncrementalPCA do
  @moduledoc """
  Incremental Principal Component Analysis.

  Description goes here (elaborate on incremental approach)

  References:

  * [1] [Incremental Learning for Robust Visual Tracking](https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf)
  """
  import Nx.Defn
  import Scholar.Shared
  require Nx

  @derive {Nx.Container,
           keep: [:num_components, :whiten?],
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
    :num_components,
    :num_samples_seen,
    :components,
    :singular_values,
    :mean,
    :variance,
    :explained_variance,
    :explained_variance_ratio,
    :whiten?
  ]

  stream_opts = [
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

  tensor_opts =
    stream_opts ++
      [
        batch_size: [
          type: :pos_integer,
          doc: "The number of samples in a batch."
        ]
      ]

  @stream_schema NimbleOptions.new!(stream_opts)
  @tensor_schema NimbleOptions.new!(tensor_opts)

  @doc """
  Fits an Incremental PCA model.

  ## Options

  #{NimbleOptions.docs(@tensor_schema)}
  """
  deftransform fit(%Nx.Tensor{} = x, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected input tensor to have shape {num_samples, num_features},
             got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    opts = NimbleOptions.validate!(opts, @tensor_schema)

    {num_samples, num_features} = Nx.shape(x)

    batch_size =
      if opts[:batch_size] do
        opts[:batch_size]
      else
        5 * num_features
      end

    num_components = opts[:num_components]

    cond do
      num_components > batch_size ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              batch_size = #{batch_size}, got #{num_components}
              """

      num_components > num_samples ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_samples = #{num_samples}, got #{num_components}
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

    opts = Keyword.put(opts, :batch_size, batch_size)

    fit_n(x, opts)
  end

  defn fit_n(x, opts) do
    batch_size = opts[:batch_size]

    {batches, leftover} =
      get_batches(x, batch_size: batch_size, min_batch_size: opts[:num_components])

    num_batches = Nx.axis_size(batches, 0)
    model = fit_first_n(batches[0], opts)

    {model, _} =
      while {
              model,
              {
                batches,
                i = Nx.u64(1)
              }
            },
            i < num_batches do
        batch = batches[i]
        model = partial_fit_n(model, batch)
        {model, {batches, i + 1}}
      end

    model =
      case leftover do
        nil -> model
        _ -> partial_fit_n(model, leftover)
      end

    model
  end

  @doc """
  Fits an Incremental PCA model on a stream of batches.

  ## Options

  #{NimbleOptions.docs(@stream_schema)}
  """
  def fit(batches = %Stream{}, opts) do
    opts = NimbleOptions.validate!(opts, @stream_schema)
    first_batch = Enum.at(batches, 0)
    model = fit_first_n(first_batch, opts)
    batches = Stream.drop(batches, 1)

    Enum.reduce(
      batches,
      model,
      # TODO: JIT
      fn batch, model -> partial_fit(model, batch) end
    )
  end

  deftransformp validate_batch(batch, opts) do
    {batch_size, num_features} = Nx.shape(batch)
    num_components = opts[:num_components]

    cond do
      num_components > batch_size ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              batch_size = #{batch_size}, got #{num_components}
              """

      num_components > num_features ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_features = #{num_features}, got #{num_components}
              """
    end
  end

  defnp fit_first_n(x, opts) do
    # This is similar to Scholar.Decomposition.PCA.fit_n
    num_samples = Nx.u64(Nx.axis_size(x, 0))
    num_components = opts[:num_components]
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
      num_samples_seen: num_samples,
      num_components: num_components,
      components: components,
      singular_values: singular_values,
      mean: mean,
      variance: variance,
      explained_variance: explained_variance[[0..(num_components - 1)]],
      explained_variance_ratio: explained_variance_ratio,
      whiten?: opts[:whiten?]
    }
  end

  @doc """
  Updates an Incremental PCA model on samples `x`.
  """
  deftransform partial_fit(model, x) do
    {num_samples, num_features} = Nx.shape(x)
    num_components = model.num_components

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

    partial_fit_n(model, x)
  end

  defnp partial_fit_n(model, x) do
    num_components = model.num_components
    components = model.components
    singular_values = model.singular_values
    num_samples_seen = model.num_samples_seen
    mean = model.mean
    variance = model.variance
    {num_samples, _} = Nx.shape(x)

    {x_centered, x_mean, new_num_samples_seen, new_mean, new_variance} =
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
      num_components: num_components,
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
    {x_centered, x_mean, new_num_samples_seen, new_mean, new_variance}
  end

  @doc """
  Documentation goes here.
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
    # This is the same as Scholar.Decomposition.PCA.transform_n!
    z = Nx.dot(x - mean, [1], components, [1])

    if whiten? do
      z / Nx.sqrt(explained_variance)
    else
      z
    end
  end
end
