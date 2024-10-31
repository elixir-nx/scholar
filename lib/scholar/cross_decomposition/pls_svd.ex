defmodule Scholar.CrossDecomposition.PLSSVD do
  @moduledoc """
  Partial Least Square SVD.

  This transformer simply performs a SVD on the cross-covariance matrix.
  It is able to project both the training data `x` and the targets
  `y`. The training data `x` is projected on the left singular vectors, while
  the targets are projected on the right singular vectors.
  """
  import Nx.Defn

  @derive {Nx.Container,
           containers: [
             :x_mean,
             :y_mean,
             :x_std,
             :y_std,
             :x_weights,
             :y_weights
           ]}
  defstruct [
    :x_mean,
    :y_mean,
    :x_std,
    :y_std,
    :x_weights,
    :y_weights
  ]

  opts_schema = [
    num_components: [
      default: 2,
      type: :pos_integer,
      doc: "The number of components to keep. Should be in `[1,
        min(n_samples, n_features, n_targets)]`."
    ],
    scale: [
      default: true,
      type: :boolean,
      doc: "Whether to scale `x` and `y`."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts_schema)

  @doc """
    Fit model to data.
    Takes as arguments: 

    * `x` - training samples, `{num_samples, num_features}` shaped tensor
    
    * `y` - targets, `{num_samples, num_targets}` shaped `y` tensor

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns fitted estimator represented by struct with the following parameters:

    * `:x_mean` - tensor of shape `{num_features}` which represents `x` tensor mean values calculated along axis 0.

    * `:y_mean` - tensor of shape `{num_targets}` which represents `x` tensor mean values calculated along axis 0.

    * `:x_std` - tensor of shape `{num_features}` which represents `x` tensor standard deviation values calculated along axis 0.

    * `:y_std` -  tensor of shape `{num_targets}` which represents `y` tensor standard deviation values calculated along axis 0.

    * `:x_weights` -  tensor of shape `{num_features, num_components}` the left singular vectors of the SVD of the cross-covariance matrix.

    * `:y_weights` -  tensor of shape `{num_targets, num_components}` the right singular vectors of the SVD of the cross-covariance matrix.

  ## Examples

      iex> x = Nx.tensor([[0.0, 0.0, 1.0],
      ...>                [1.0, 0.0, 0.0],
      ...>                [2.0, 2.0, 2.0],
      ...>                [2.0, 5.0, 4.0]])
      iex> y = Nx.tensor([[0.1, -0.2],
      ...>                [0.9, 1.1],
      ...>                [6.2, 5.9],
      ...>                [11.9, 12.3]])
      iex> model = Scholar.CrossDecomposition.PLSSVD.fit(x, y)
      iex> model.x_mean
      #Nx.Tensor<
        f32[3]
        [1.25, 1.75, 1.75]
      >
      iex> model.y_std
      #Nx.Tensor<
        f32[2]
        [5.467098712921143, 5.661198616027832]
      >
      iex> model.x_weights
      #Nx.Tensor<
        f32[3][2]
        [
          [0.521888256072998, -0.11256571859121323],
          [0.6170258522033691, 0.7342619299888611],
          [0.5889922380447388, -0.6694686412811279]
        ]
      >
  """

  deftransform fit(x, y, opts \\ []) do
    fit_n(x, y, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_n(x, y, opts) do
    {x, y} = check_x_y(x, y, opts)
    num_components = opts[:num_components]
    {x, y, x_mean, y_mean, x_std, y_std} = center_scale_x_y(x, y, opts)

    c =
      Nx.transpose(x)
      |> Nx.dot(y)

    {u, _s, vt} = Nx.LinAlg.svd(c, full_matrices?: false)
    u = Nx.slice_along_axis(u, 0, num_components, axis: 1)
    vt = Nx.slice_along_axis(vt, 0, num_components, axis: 0)
    {u, vt} = Scholar.Decomposition.Utils.flip_svd(u, vt)
    v = Nx.transpose(vt)

    x_weights = u
    y_weights = v

    %__MODULE__{
      x_mean: x_mean,
      y_mean: y_mean,
      x_std: x_std,
      y_std: y_std,
      x_weights: x_weights,
      y_weights: y_weights
    }
  end

  @doc """
    Apply the dimensionality reduction.
    Takes as arguments: 

    * fitted estimator struct which is return value of `fit/3` function from this module

    * `x` - training samples, `{num_samples, num_features}` shaped tensor
    
    * `y` - targets, `{num_samples, num_targets}` shaped `y` tensor

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values
    
    Returns tuple with transformed data `{x_transformed, y_transformed}` where:

    * `x_transformed` is `{num_samples, num_features}` shaped tensor.
    
    * `y_transformed` is `{num_samples, num_features}` shaped tensor.

  ## Examples

      iex> x = Nx.tensor([[0.0, 0.0, 1.0],
      ...>                [1.0, 0.0, 0.0],
      ...>                [2.0, 2.0, 2.0],
      ...>                [2.0, 5.0, 4.0]])
      iex> y = Nx.tensor([[0.1, -0.2],
      ...>                [0.9, 1.1],
      ...>                [6.2, 5.9],
      ...>                [11.9, 12.3]])
      iex> model = Scholar.CrossDecomposition.PLSSVD.fit(x, y)
      iex> {x, y} = Scholar.CrossDecomposition.PLSSVD.transform(model, x, y)
      iex> x
      #Nx.Tensor<
        f32[4][2]
        [
          [-1.397004246711731, -0.10283949971199036],
          [-1.1967883110046387, 0.17159013450145721],
          [0.5603229403495789, -0.10849219560623169],
          [2.0334696769714355, 0.039741579443216324]
        ]
      >
      iex> y
      #Nx.Tensor<
        f32[4][2]
        [
          [-1.2260178327560425, -0.019306711852550507],
          [-0.9602956175804138, 0.04015407711267471],
          [0.3249155580997467, -0.04311027377843857],
          [1.8613981008529663, 0.022262824699282646]
        ]
      >

  """
  deftransform transform(model, x, y, opts \\ []) do
    transform_n(model, x, y, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp transform_n(
          %__MODULE__{
            x_mean: x_mean,
            y_mean: y_mean,
            x_std: x_std,
            y_std: y_std,
            x_weights: x_weights,
            y_weights: y_weights
          } = _model,
          x,
          y,
          opts
        ) do
    {x, y} = check_x_y(x, y, opts)

    xr = (x - x_mean) / x_std
    x_scores = Nx.dot(xr, x_weights)

    yr = (y - y_mean) / y_std
    y_scores = Nx.dot(yr, y_weights)
    {x_scores, y_scores}
  end

  @doc """
  Learn and apply the dimensionality reduction.

  The arguments are:
    
    * `x` - training samples, `{num_samples, num_features}` shaped tensor
    
    * `y` - targets, `{num_samples, num_targets}` shaped `y` tensor

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  Returns tuple with transformed data `{x_transformed, y_transformed}` where:

    * `x_transformed` is `{num_samples, num_features}` shaped tensor.
    
    * `y_transformed` is `{num_samples, num_features}` shaped tensor.

  ## Examples

      iex> x = Nx.tensor([[0.0, 0.0, 1.0],
      ...>                [1.0, 0.0, 0.0],
      ...>                [2.0, 2.0, 2.0],
      ...>                [2.0, 5.0, 4.0]])
      iex> y = Nx.tensor([[0.1, -0.2],
      ...>                [0.9, 1.1],
      ...>                [6.2, 5.9],
      ...>                [11.9, 12.3]])
      iex> {x, y} = Scholar.CrossDecomposition.PLSSVD.fit_transform(x, y)
      iex> x
      #Nx.Tensor<
        f32[4][2]
        [
          [-1.397004246711731, -0.10283949971199036],
          [-1.1967883110046387, 0.17159013450145721],
          [0.5603229403495789, -0.10849219560623169],
          [2.0334696769714355, 0.039741579443216324]
        ]
      >
      iex> y
      #Nx.Tensor<
        f32[4][2]
        [
          [-1.2260178327560425, -0.019306711852550507],
          [-0.9602956175804138, 0.04015407711267471],
          [0.3249155580997467, -0.04311027377843857],
          [1.8613981008529663, 0.022262824699282646]
        ]
      >

  """

  deftransform fit_transform(x, y, opts \\ []) do
    fit_transform_n(x, y, NimbleOptions.validate!(opts, @opts_schema))
  end

  defnp fit_transform_n(x, y, opts) do
    fit(x, y, opts)
    |> transform(x, y, opts)
  end

  defnp check_x_y(x, y, opts) do
    y =
      case Nx.shape(y) do
        {n} -> Nx.reshape(y, {n, 1})
        _ -> y
      end

    num_components = opts[:num_components]
    {num_samples, num_features} = Nx.shape(x)
    {num_samples_y, num_targets} = Nx.shape(y)

    cond do
      num_samples != num_samples_y ->
        raise ArgumentError,
              """
              num_samples must be the same for x and y \
              x num_samples = #{num_samples}, y num_samples = #{num_samples_y}
              """

      num_components > num_features ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_features = #{num_features}, got #{num_components}
              """

      num_components > num_samples ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_samples = #{num_samples}, got #{num_components}
              """

      num_components > num_targets ->
        raise ArgumentError,
              """
              num_components must be less than or equal to \
              num_targets = #{num_targets}, got #{num_components}
              """

      true ->
        nil
    end

    {x, y}
  end

  defnp center_scale_x_y(x, y, opts) do
    scale = opts[:scale]
    x_mean = Nx.mean(x, axes: [0])
    x = x - x_mean

    y_mean = Nx.mean(y, axes: [0])
    y = y - y_mean

    if scale do
      x_std = Nx.standard_deviation(x, axes: [0], ddof: 1)
      x_std = Nx.select(x_std == 0.0, 1.0, x_std)
      x = x / Nx.broadcast(x_std, Nx.shape(x))

      y_std = Nx.standard_deviation(y, axes: [0], ddof: 1)
      y_std = Nx.select(y_std == 0.0, 1.0, y_std)
      y = y / Nx.broadcast(y_std, Nx.shape(y))

      {x, y, x_mean, y_mean, x_std, y_std}
    else
      x_std = Nx.broadcast(1, {Nx.axis_size(x, 1)})
      y_std = Nx.broadcast(1, {Nx.axis_size(y, 1)})
      {x, y, x_mean, y_mean, x_std, y_std}
    end
  end
end
