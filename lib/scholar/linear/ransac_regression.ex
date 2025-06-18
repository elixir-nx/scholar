defmodule RANSACRegression do
  @moduledoc """
  The Random Sample Consensus algorithm is an iterative
  method that fits a robust estimator, able to cope with
  a large proportion of outliers in the data.

  ## References

  [Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography](https://www.cs.ait.ac.th/~mdailey/cvreadings/Fischler-RANSAC.pdf)
  """

  import Nx.Defn
  alias Scholar.Metrics.Regression, as: Metrics

  alias Scholar.Linear.{
    LinearRegression,
    PolynomialRegression
  }

  @derive {Nx.Container, containers: [:model, :inliers_indices, :n_inliers]}
  defstruct [:estimator, :model, :inliers_indices, :n_inliers]

  @models [linear: 1, polynomial: 2]
  @losses [mae: 1, mse: 2]

  opts = [
    estimator: [
      type: {:in, Keyword.keys(@models)},
      doc: """
      Base estimator.
      """,
      default: :linear
    ],
    max_iters: [
      type: :integer,
      doc: """
      Maximum number of iterations for random sample selection.
      """,
      default: 100
    ],
    threshold: [
      type: :float,
      required: true,
      doc: """
      Maximum error for a sample to be considered classified as inlier.
      """
    ],
    min_samples: [
      type: :integer,
      required: true,
      doc: """
      Minimum number of samples chosen randomly from original data.
      """
    ],
    degree: [
      type: :integer,
      doc: """
      Degree for a polynomial regression base estimator. Default is 2.
      """,
      default: 2
    ],
    loss: [
      type: {:in, Keyword.keys(@losses)},
      default: :mae,
      doc: """
      Loss function to evaluate estimator. If the loss in a sample is strictly
      lesser than `threshold`, then this sample is classified as an inlier.
      """
    ],
    random_seed: [
      type: :integer,
      default: 42
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a robust regressor using RANSAC algorithm.

  #{NimbleOptions.docs(@opts_schema)}
  """
  def fit(x, y, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    inliers_mask = fit_n(x, y, opts)
    n_inliers = Nx.to_number(Nx.sum(inliers_mask))

    if n_inliers == 0 do
      raise "RANSAC was not able to find consensus set that
            meets the required criteria."
    end

    inliers_idx = Nx.argsort(inliers_mask, direction: :desc)[0..(n_inliers - 1)]

    x_inliers = Nx.take(x, inliers_idx)
    y_inliers = Nx.take(y, inliers_idx)

    model =
      cond do
        opts[:estimator] == :linear ->
          LinearRegression.fit(x_inliers, y_inliers)

        opts[:estimator] == :polynomial ->
          PolynomialRegression.fit(x_inliers, y_inliers)

        true ->
          LinearRegression.fit(x_inliers, y_inliers)
      end

    %__MODULE__{
      estimator: opts[:estimator],
      inliers_indices: inliers_idx,
      n_inliers: n_inliers,
      model: model
    }
  end

  def predict(%__MODULE__{estimator: :polynomial, model: m}, x),
    do: PolynomialRegression.predict(m, x)

  def predict(%__MODULE__{model: m}, x), do: LinearRegression.predict(m, x)

  defnp loss_fn(loss_t, y_true, y_pred) do
    {loss} = loss_t.shape

    cond do
      loss == @losses[:mae] -> Metrics.mean_absolute_error(y_true, y_pred, axes: [1])
      loss == @losses[:mse] -> Metrics.mean_square_error(y_true, y_pred, axes: [1])
      true -> Metrics.mean_absolute_error(y_true, y_pred, axis: 0)
    end
  end

  defnp fit_fn(model_t, x_train, y_train, degree_t) do
    {m} = model_t.shape
    {d} = degree_t.shape

    cond do
      m == @models[:linear] ->
        LinearRegression.fit(x_train, y_train)

      m == @models[:polynomial] ->
        PolynomialRegression.fit(x_train, y_train, degree: d)

      true ->
        LinearRegression.fit(x_train, y_train)
    end
  end

  defnp predict_fn(model_t, model, x) do
    {m} = model_t.shape

    cond do
      m == @models[:linear] ->
        LinearRegression.predict(model, x)

      m == @models[:polynomial] ->
        PolynomialRegression.predict(model, x)

      true ->
        LinearRegression.predict(model, x)
    end
  end

  defnp fit_n(x, y, opts) do
    max_iters = opts[:max_iters]
    thr = opts[:threshold]
    min_samples = opts[:min_samples]
    loss = @losses[opts[:loss]]
    model = @models[opts[:estimator]]
    degree = opts[:degree]

    loss_t = Nx.broadcast(:nan, {loss})
    model_t = Nx.broadcast(:nan, {model})
    min_samples_t = Nx.broadcast(:nan, {min_samples})
    degree_t = Nx.broadcast(:nan, {degree})

    rand_key = Nx.Random.key(opts[:random_seed])
    data = Nx.concatenate([x, y], axis: 1)
    inliers = Nx.broadcast(0, {max_iters, elem(x.shape, 0)})

    {inliers_masks, _} =
      while {inliers,
             {i = 0, x, y, rand_key, data, min_samples_t, thr, loss_t, model_t, degree_t}},
            i < max_iters do
        n_samples = elem(min_samples_t.shape, 0)
        {rand_samples, rand_key} = Nx.Random.choice(rand_key, data, axis: 0, samples: n_samples)

        {rand_x, rand_y} = Nx.split(rand_samples, elem(x.shape, 1), axis: 1)
        model = fit_fn(model_t, rand_x, rand_y, degree_t)
        y_pred = predict_fn(model_t, model, x)
        y_pred = Nx.reshape(y_pred, {elem(y_pred.shape, 0), 1})

        error = loss_fn(loss_t, y, y_pred)
        inliers_i = Nx.less(error, thr)

        updated_inliers =
          Nx.put_slice(inliers, [i, 0], Nx.reshape(inliers_i, {1, elem(x.shape, 0)}))

        {updated_inliers,
         {i + 1, x, y, rand_key, data, min_samples_t, thr, loss_t, model_t, degree_t}}
      end

    best_mask_idx =
      inliers_masks
      |> Nx.vectorize(:masks)
      |> Nx.sum()
      |> Nx.devectorize(keep_names: false)
      |> Nx.argmax()

    inliers_masks[best_mask_idx]
  end
end
