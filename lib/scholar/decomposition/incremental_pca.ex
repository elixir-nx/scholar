defmodule Scholar.Decomposition.IncrementalPCA do
  @moduledoc """
  Incremental Principal Component Analysis.

  References:

  * [1] - [Incremental Learning for Robust Visual Tracking](https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf)
  """
  import Nx.Defn
  import Scholar.Shared
  require Nx

  @derive {Nx.Container,
           keep: [:num_samples_seen, :num_components],
           containers: [
             :components,
             :singular_values,
             :mean,
             :variance,
             :explained_variance,
             :explained_variance_ratio
           ]}
  defstruct [
    :num_samples_seen,
    :num_components,
    :components,
    :singular_values,
    :mean,
    :variance,
    :explained_variance,
    :explained_variance_ratio
  ]

  opts = [
    num_components: [
      type: :pos_integer,
      doc: "???"
    ],
    batch_size: [
      type: :pos_integer,
      doc: "The number of samples in a batch."
    ],
    whiten: [
      type: :boolean,
      default: false,
      doc: "???"
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  deftransform fit(x, opts) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected input tensor to have shape {num_samples, num_features},
             got tensor with shape: #{inspect(Nx.shape(x))}"
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)

    {num_samples, num_features} = Nx.shape(x)

    batch_size =
      if opts[:batch_size] do
        opts[:batch_size]
      else
        5 * num_features
      end

    # TODO: What to do if batch_size is greater than num_samples? Probably raise an error.

    num_components = opts[:num_components]

    num_components =
      if num_components do
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
            num_components
        end
      else
        Enum.min([batch_size, num_samples, num_features])
      end

    opts = Keyword.put(opts, :num_components, num_components)
    opts = Keyword.put(opts, :batch_size, batch_size)

    fit_n(x, opts)
  end

  defn fit_n(x, opts) do
    batch_size = opts[:batch_size]
    {batches, leftover} = get_batches(x, batch_size: batch_size)

    num_batches = Nx.axis_size(batches, 0)
    model = fit_first(batches[0], opts)

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

    # partial_fit_n(model, leftover)
    model
  end

  defnp fit_first(x, opts) do
    # This is similar to Scholar.Decomposition.PCA.fit_n
    {num_samples, _} = Nx.shape(x)
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
      explained_variance_ratio: explained_variance_ratio
    }
  end

  defnp partial_fit_n(model, x) do
    num_samples_seen = model.num_samples_seen
    num_components = model.num_components
    components = model.components
    singular_values = model.singular_values
    mean = model.mean
    variance = model.variance
    {num_samples, _} = Nx.shape(x)
    last_sample_count = Nx.broadcast(num_samples_seen, {Nx.axis_size(x, 1)})

    {num_total_samples, col_mean, col_variance} =
      incremental_mean_and_var(x, last_sample_count, mean, variance)

    num_total_samples = num_total_samples[0]
    # if num_samples_seen > 0 do
    col_batch_mean = Nx.mean(x, axes: [0])
    x_centered = x - col_batch_mean

    mean_correction =
      Nx.sqrt(num_samples_seen / num_total_samples) * num_samples * (mean - col_batch_mean)

    mean_correction = Nx.new_axis(mean_correction, 0)
    to_add = Nx.reshape(singular_values, {1, :auto}) * components
    z = Nx.concatenate([Nx.transpose(to_add), x_centered, mean_correction], axis: 0)
    {u, s, vt} = Nx.LinAlg.svd(z, full_matrices?: false)
    {_, vt} = Scholar.Decomposition.Utils.flip_svd(u, vt)
    components = vt[[0..(num_components - 1)]]
    singular_values = s[[0..(num_components - 1)]]
    explained_variance = s * s / (num_total_samples - 1)

    explained_variance_ratio =
      singular_values * singular_values / Nx.sum(col_variance * num_total_samples)

    %__MODULE__{
      num_samples_seen: num_total_samples,
      num_components: num_components,
      components: components,
      singular_values: singular_values,
      mean: col_mean,
      variance: col_variance,
      explained_variance: explained_variance,
      explained_variance_ratio: explained_variance_ratio
    }
  end

  defnp incremental_mean_and_var(x, last_sample_count, last_mean, last_variance) do
    new_sample_count = Nx.axis_size(x, 0)
    updated_sample_count = last_sample_count + new_sample_count
    last_sum = last_sample_count * last_mean
    new_sum = Nx.sum(x, axes: [0])
    updated_mean = (last_sum + new_sum) / updated_sample_count
    t = new_sum / new_sample_count
    temp = x - t
    correction = Nx.sum(temp, axes: [0])
    temp = temp * temp

    new_unnormalized_variance =
      Nx.sum(temp, axes: [0]) - correction * correction / new_sample_count

    last_unnormalized_variance = last_sample_count * last_variance
    last_over_new_count = last_sample_count / new_sample_count

    updated_unnormalized_variance =
      last_unnormalized_variance +
        new_unnormalized_variance +
        last_over_new_count / updated_sample_count *
          (last_sum / last_over_new_count - new_sum) ** 2

    zeros = last_sample_count == 0

    updated_unnormalized_variance =
      Nx.select(zeros, new_unnormalized_variance, updated_unnormalized_variance)

    updated_variance = updated_unnormalized_variance / updated_sample_count
    {updated_sample_count, updated_mean, updated_variance}
  end
end
