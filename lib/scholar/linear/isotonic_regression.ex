defmodule Scholar.Linear.IsotonicRegression do
  @moduledoc """
  Isotonic regression is a method of fitting a free-form line to a set of
  observations by solving a convex optimization problem. It is a form of
  regression analysis that can be used as an alternative to polynomial
  regression to fit nonlinear data.

  Time complexity of isotonic regression is $O(N^2)$ where $N$ is the
  number of points.
  """
  require Nx
  import Nx.Defn, except: [transform: 2]
  import Scholar.Shared
  alias Scholar.Linear.LinearHelpers

  @derive {
    Nx.Container,
    containers: [
      :increasing,
      :x_min,
      :x_max,
      :x_thresholds,
      :y_thresholds,
      :cutoff_index,
      :preprocess
    ]
  }
  defstruct [
    :x_min,
    :x_max,
    :x_thresholds,
    :y_thresholds,
    :increasing,
    :cutoff_index,
    :preprocess
  ]

  @type t() :: %__MODULE__{
          x_min: Nx.Tensor.t(),
          x_max: Nx.Tensor.t(),
          x_thresholds: Nx.Tensor.t(),
          y_thresholds: Nx.Tensor.t(),
          increasing: Nx.Tensor.t(),
          cutoff_index: Nx.Tensor.t(),
          preprocess: tuple() | Scholar.Interpolation.Linear.t()
        }

  opts = [
    y_min: [
      type: :float,
      doc: """
      Lower bound on the lowest predicted value. If if not provided, the lower bound
      is set to `Nx.Constant.neg_infinity()`.
      """
    ],
    y_max: [
      type: :float,
      doc: """
      Upper bound on the highest predicted value. If if not provided, the lower bound
      is set to `Nx.Constant.infinity()`.
      """
    ],
    increasing: [
      type: {:in, [:auto, true, false]},
      default: :auto,
      doc: """
      Whether the isotonic regression should be fit with the constraint that the
      function is monotonically increasing. If `false`, the constraint is that
      the function is monotonically decreasing. If `:auto`, the constraint is
      determined automatically based on the data.
      """
    ],
    out_of_bounds: [
      type: {:in, [:clip, :nan]},
      default: :nan,
      doc: """
      How to handle out-of-bounds points. If `:clip`, out-of-bounds points are
      mapped to the nearest valid value. If `:nan`, out-of-bounds points are
      replaced with `Nx.Constant.nan()`.
      """
    ],
    sample_weights: [
      type: {:custom, Scholar.Options, :weights, []},
      doc: """
      The weights for each observation. If not provided,
      all observations are assigned equal weight.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a isotonic regression model for sample inputs `x` and
  sample targets `y`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:x_min` - Minimum value of input tensor `x`.

    * `:x_max` - Maximum value of input tensor `x`.

    * `:x_thresholds` - Thresholds used for predictions.

    * `:y_thresholds` - Predicted values associated with each threshold.

    * `:increasing` - Whether the isotonic regression is increasing.

    * `:cutoff_index` - The index of the last valid threshold. Rest elements are placeholders
      for the sake of preserving shape of tensor.

    * `:preprocess` - Interpolation function to be applied on input tensor `x`. Before `preprocess/1`
      is applied it is set to {}

  ## Examples

      iex> x = Nx.tensor([1, 4, 7, 9, 10, 11])
      iex> y = Nx.tensor([1, 3, 6, 8, 9, 10])
      iex> Scholar.Linear.IsotonicRegression.fit(x, y)
      %Scholar.Linear.IsotonicRegression{
        x_min: Nx.tensor(
          1.0
        ),
        x_max: Nx.tensor(
          11.0
        ),
        x_thresholds: Nx.tensor(
          [1.0, 4.0, 7.0, 9.0, 10.0, 11.0]
        ),
        y_thresholds: Nx.tensor(
          [1.0, 3.0, 6.0, 8.0, 9.0, 10.0]
        ),
        increasing: Nx.u8(1),
        cutoff_index: Nx.tensor(
          5
        ),
        preprocess: {}
      }
  """
  deftransform fit(x, y, opts \\ []) do
    {n_samples} = Nx.shape(x)
    y = LinearHelpers.validate_y_shape(y, n_samples, __MODULE__)

    opts = NimbleOptions.validate!(opts, @opts_schema)

    opts =
      [
        sample_weights_flag: opts[:sample_weights] != nil
      ] ++
        opts

    {sample_weights, opts} = Keyword.pop(opts, :sample_weights, 1.0)
    x_type = to_float_type(x)
    x = to_float(x)
    y = to_float(y)

    sample_weights =
      if Nx.is_tensor(sample_weights),
        do: Nx.as_type(sample_weights, x_type),
        else: Nx.tensor(sample_weights, type: x_type)

    sample_weights = Nx.broadcast(sample_weights, {Nx.axis_size(y, 0)})

    {increasing, opts} = Keyword.pop(opts, :increasing)

    increasing =
      case increasing do
        :auto ->
          check_increasing(x, y)

        true ->
          Nx.u8(1)

        false ->
          Nx.u8(0)
      end

    fit_n(x, y, sample_weights, increasing, opts)
  end

  defnp fit_n(x, y, sample_weights, increasing, opts) do
    {x_min, x_max, x_unique, y, index_cut} = build_y(x, y, sample_weights, increasing, opts)

    %__MODULE__{
      x_min: x_min,
      x_max: x_max,
      x_thresholds: x_unique,
      y_thresholds: y,
      increasing: increasing,
      cutoff_index: index_cut,
      preprocess: {}
    }
  end

  @doc """
  Makes predictions with the given `model` on input `x` and interpolating `function`.

  Output predictions have shape `{n_samples}` when train target is shaped either `{n_samples}` or `{n_samples, 1}`.
  Otherwise, predictions match train target shape.

  ## Examples

      iex> x = Nx.tensor([1, 4, 7, 9, 10, 11])
      iex> y = Nx.tensor([1, 3, 6, 8, 9, 10])
      iex> model = Scholar.Linear.IsotonicRegression.fit(x, y)
      iex> model = Scholar.Linear.IsotonicRegression.preprocess(model)
      iex> to_predict = Nx.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
      iex> Scholar.Linear.IsotonicRegression.predict(model, to_predict)
      #Nx.Tensor<
        f32[10]
        [1.0, 1.6666667461395264, 2.3333332538604736, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
      >
  """
  defn predict(model, x) do
    check_input_shape(x)
    check_preprocess(model)

    x = Nx.flatten(x)
    x = Nx.clip(x, model.x_min, model.x_max)

    Scholar.Interpolation.Linear.predict(
      model.preprocess,
      x
    )
  end

  @doc """
  Preprocesses the `model` for prediction.

  Returns an updated `model`.

  ## Examples

      iex> x = Nx.tensor([1, 4, 7, 9, 10, 11])
      iex> y = Nx.tensor([1, 3, 6, 8, 9, 10])
      iex> model = Scholar.Linear.IsotonicRegression.fit(x, y)
      iex> Scholar.Linear.IsotonicRegression.preprocess(model)
      %Scholar.Linear.IsotonicRegression{
        x_min: Nx.tensor(
          1.0
        ),
        x_max: Nx.tensor(
          11.0
        ),
        x_thresholds: Nx.tensor(
          [1.0, 4.0, 7.0, 9.0, 10.0, 11.0]
        ),
        y_thresholds: Nx.tensor(
          [1.0, 3.0, 6.0, 8.0, 9.0, 10.0]
        ),
        increasing: Nx.u8(1),
        cutoff_index: Nx.tensor(
          5
        ),
        preprocess: %Scholar.Interpolation.Linear{
          coefficients: Nx.tensor(
            [
              [0.6666666865348816, 0.3333333134651184],
              [1.0, -1.0],
              [1.0, -1.0],
              [1.0, -1.0],
              [1.0, -1.0]
            ]
          ),
          x: Nx.tensor(
            [1.0, 4.0, 7.0, 9.0, 10.0, 11.0]
          )
        }
      }
  """
  def preprocess(%__MODULE__{} = model, trim_duplicates \\ true) do
    cutoff = Nx.to_number(model.cutoff_index)
    x = model.x_thresholds[0..cutoff]
    y = model.y_thresholds[0..cutoff]

    {x, y} =
      if trim_duplicates do
        keep_mask =
          Nx.logical_or(
            Nx.not_equal(y[1..-2//1], y[0..-3//1]),
            Nx.not_equal(y[1..-2//1], y[2..-1//1])
          )

        keep_mask = Nx.concatenate([Nx.tensor([1]), keep_mask, Nx.tensor([1])])

        indices =
          Nx.iota({Nx.axis_size(y, 0)})
          |> Nx.add(1)
          |> Nx.multiply(keep_mask)
          |> Nx.to_flat_list()

        indices = Enum.filter(indices, fn x -> x != 0 end) |> Nx.tensor() |> Nx.subtract(1)
        x = Nx.take(x, indices)
        y = Nx.take(y, indices)
        {x, y}
      else
        {x, y}
      end

    %__MODULE__{
      model
      | x_thresholds: x,
        y_thresholds: y,
        preprocess: Scholar.Interpolation.Linear.fit(x, y)
    }
  end

  deftransform check_preprocess(model) do
    if model.preprocess == {} do
      raise ArgumentError,
            "model has not been preprocessed. " <>
              "Please call preprocess/1 on the model before calling predict/2"
    end
  end

  defnp lexsort(x, y) do
    iota = Nx.iota(Nx.shape(x))
    indices = Nx.argsort(x)
    y = Nx.take(y, indices)
    iota = Nx.take(iota, indices)
    indices = Nx.argsort(y)
    Nx.take(iota, indices)
  end

  defnp build_y(x, y, sample_weights, increasing, opts) do
    check_input_shape(x)
    x = Nx.flatten(x)
    lex_indices = lexsort(y, x)
    x = Nx.take(x, lex_indices)
    y = Nx.take(y, lex_indices)
    sample_weights = Nx.take(sample_weights, lex_indices)

    {x_unique, y_unique, sample_weights_unique, index_cut} = make_unique(x, y, sample_weights)

    y = isotonic_regression(y_unique, sample_weights_unique, index_cut, increasing, opts)

    x_min =
      Nx.reduce_min(
        Nx.select(Nx.iota(Nx.shape(x_unique)) <= index_cut, x_unique, Nx.Constants.infinity())
      )

    x_max =
      Nx.reduce_max(
        Nx.select(Nx.iota(Nx.shape(x_unique)) <= index_cut, x_unique, Nx.Constants.neg_infinity())
      )

    {x_min, x_max, x_unique, y, index_cut}
  end

  defnp isotonic_regression(y, sample_weights, max_size, increasing, opts) do
    y_min =
      case opts[:y_min] do
        nil -> Nx.Constants.neg_infinity()
        _ -> opts[:y_min]
      end

    y_max =
      case opts[:y_max] do
        nil -> Nx.Constants.infinity()
        _ -> opts[:y_max]
      end

    y = contiguous_isotonic_regression(y, sample_weights, max_size, increasing)

    Nx.clip(y, y_min, y_max)
  end

  deftransformp check_input_shape(x) do
    if not (Nx.rank(x) == 1 or (Nx.rank(x) == 2 and Nx.axis_size(x, 1) == 1)) do
      raise ArgumentError,
            "Expected input to be a 1d tensor or 2d tensor with axis 1 of size 1, " <>
              "got: #{inspect(Nx.shape(x))}"
    end
  end

  defnp make_unique(x, y, sample_weights) do
    x_output = Nx.broadcast(Nx.tensor(0, type: Nx.type(x)), x)

    sample_weights_output =
      Nx.broadcast(Nx.tensor(1, type: Nx.type(sample_weights)), sample_weights)

    type_wy = Nx.Type.merge(Nx.type(y), Nx.type(sample_weights_output))
    y_output = Nx.broadcast(Nx.tensor(0, type: type_wy), y)

    current_x = Nx.as_type(x[0], Nx.type(x))
    current_y = Nx.tensor(0, type: type_wy)
    current_weight = Nx.tensor(0, type: Nx.type(sample_weights))

    index = 0

    {{x_output, y_output, sample_weights_output, index, current_x, current_y, current_weight}, _} =
      while {{x_output, y_output, sample_weights_output, index, current_x, current_y,
              current_weight}, {j = 0, eps = 1.0e-10, y, x, sample_weights}},
            j < Nx.axis_size(x, 0) do
        x_j = x[j]

        {x_output, y_output, sample_weights_output, index, current_x, current_weight, current_y} =
          if x_j - current_x >= eps do
            x_output = Nx.indexed_put(x_output, Nx.new_axis(index, 0), current_x)
            y_output = Nx.indexed_put(y_output, Nx.new_axis(index, 0), current_y / current_weight)

            sample_weights_output =
              Nx.indexed_put(sample_weights_output, Nx.new_axis(index, 0), current_weight)

            index = index + 1
            current_x = x_j
            current_weight = sample_weights[j]
            current_y = y[j] * sample_weights[j]

            {x_output, y_output, sample_weights_output, index, current_x, current_weight,
             current_y}
          else
            current_weight = current_weight + sample_weights[j]
            current_y = current_y + y[j] * sample_weights[j]

            {x_output, y_output, sample_weights_output, index, current_x, current_weight,
             current_y}
          end

        {{x_output, y_output, sample_weights_output, index, current_x, current_y, current_weight},
         {j + 1, eps, y, x, sample_weights}}
      end

    x_output = Nx.indexed_put(x_output, Nx.new_axis(index, 0), current_x)
    y_output = Nx.indexed_put(y_output, Nx.new_axis(index, 0), current_y / current_weight)

    sample_weights_output =
      Nx.indexed_put(sample_weights_output, Nx.new_axis(index, 0), current_weight)

    {x_output, y_output, sample_weights_output, index}
  end

  defnp contiguous_isotonic_regression(y, sample_weights, max_size, increasing) do
    y_size = if(increasing, do: max_size, else: Nx.axis_size(y, 0) - 1) |> Nx.as_type(:u32)
    y = if increasing, do: y, else: Nx.reverse(y)
    sample_weights = if increasing, do: sample_weights, else: Nx.reverse(sample_weights)

    target = Nx.iota({Nx.axis_size(y, 0)}, type: :u32)
    type_wy = Nx.Type.merge(Nx.type(y), Nx.type(sample_weights))
    i = if(increasing, do: 0, else: Nx.axis_size(y, 0) - 1 - max_size) |> Nx.as_type(:u32)

    {{y, target}, _} =
      while {{y, target},
             {i, sample_weights, sum_w = Nx.tensor(0, type: Nx.type(sample_weights)),
              sum_wy = Nx.tensor(0, type: type_wy), prev_y = Nx.tensor(0, type: type_wy),
              _k = Nx.u32(0), terminating_flag = Nx.u8(0), y_size}},
            i < y_size + 1 and not terminating_flag do
        k = target[i] + 1

        cond do
          k == y_size + 1 ->
            {{y, target}, {i, sample_weights, sum_w, sum_wy, prev_y, k, 1, y_size}}

          y[i] < y[k] ->
            i = k

            {{y, target}, {i, sample_weights, sum_w, sum_wy, prev_y, k, terminating_flag, y_size}}

          true ->
            sum_wy = sample_weights[i] * y[i]
            sum_w = sample_weights[i]

            {y, sample_weights, i, target, sum_w, sum_wy, prev_y, k, _inner_terminating_flag,
             y_size} =
              while {y, sample_weights, i, target, sum_w, sum_wy, _prev_y = prev_y, k,
                     inner_terminating_flag = 0, y_size},
                    not inner_terminating_flag do
                prev_y = y[k]
                sum_wy = sum_wy + sample_weights[k] * y[k]
                sum_w = sum_w + sample_weights[k]
                k = target[k] + 1

                {y, sample_weights, target, i, inner_terminating_flag} =
                  if k == y_size + 1 or prev_y < y[k] do
                    y = Nx.indexed_put(y, Nx.new_axis(i, 0), sum_wy / sum_w)
                    sample_weights = Nx.indexed_put(sample_weights, Nx.new_axis(i, 0), sum_w)
                    target = Nx.indexed_put(target, Nx.new_axis(i, 0), k - 1)
                    target = Nx.indexed_put(target, Nx.new_axis(k - 1, 0), i)

                    i =
                      if i > 0 do
                        target[i - 1]
                      else
                        i
                      end

                    {y, sample_weights, target, i, 1}
                  else
                    {y, sample_weights, target, i, 0}
                  end

                {y, sample_weights, i, target, sum_w, sum_wy, prev_y, k, inner_terminating_flag,
                 y_size}
              end

            {{y, target}, {i, sample_weights, sum_w, sum_wy, prev_y, k, terminating_flag, y_size}}
        end
      end

    i = if(increasing, do: 0, else: Nx.axis_size(y, 0) - 1 - max_size) |> Nx.as_type(:u32)

    {y, _} =
      while {y, {target, i, _k = Nx.u32(0), max_size}}, i < max_size + 1 do
        k = target[i] + 1
        indices = Nx.iota({Nx.axis_size(y, 0)}, type: :u32)
        in_range? = Nx.logical_and(i + 1 <= indices, indices < k)
        y = Nx.select(in_range?, y[i], y)
        i = k
        {y, {target, i, k, max_size}}
      end

    if increasing, do: y, else: Nx.reverse(y)
  end

  defnp check_increasing(x, y) do
    x = Nx.new_axis(x, -1)
    y = Nx.new_axis(y, -1)
    model = Scholar.Linear.LinearRegression.fit(x, y)
    model.coefficients[0] >= 0
  end
end
