defmodule Scholar.Neighbors.BruteKNN do
  @moduledoc """
  Brute-Force k-Nearest Neighbor Search Algorithm.

  <how it works> .. i.e. ...
  Larger batch sizes will be faster, but will consume more memory.
  Time complexity is $O(MN)$ for $N$ samples and $M$ query points.
  Memory complexity is $O(BN)$ for batch size B.
  """
  import Nx.Defn
  import Scholar.Shared
  require Nx

  @derive {Nx.Container, keep: [:num_neighbors, :metric, :batch_size], containers: [:data]}
  defstruct [:num_neighbors, :metric, :data, :batch_size]

  # @derive {Nx.Container, keep: [:num_neighbors, :batch_size], containers: [:data]}
  # defstruct [:num_neighbors, :data, :batch_size]

  opts = [
    num_neighbors: [
      required: true,
      type: :pos_integer,
      doc: "The number of nearest neighbors."
    ],
    metric: [
      type: {:or, [{:custom, Scholar.Options, :metric, []}, {:fun, 2}]},
      default: {:minkowski, 2},
      doc: ~S"""
      The function that measures distance between two points. Possible values:

      * `{:minkowski, p}` - Minkowski metric. By changing value of `p` parameter (a positive number or `:infinity`)
      we can set Manhattan (`1`), Euclidean (`2`), Chebyshev (`:infinity`), or any arbitrary $L_p$ metric.

      * `:cosine` - Cosine metric.

      * Anonymous function of arity 2 that takes two rank-1 tensors of same dimension and returns a scalar.
      """
    ],
    batch_size: [
      type: :pos_integer,
      doc: "The number of samples in a batch."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a brute-force k-NN model.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Examples

      iex> data = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> Scholar.Neighbors.BruteKNN.fit(data, num_neighbors: 2)
      %Scholar.Neighbors.BruteKNN{
        num_neighbors: 2,
        metric: #Function<5.26796099/2 in Scholar.Neighbors.BruteKNN.fit/2>,
        data: Nx.Tensor(
          [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6]
          ]
        ),
        batch_size: nil
      }
  """
  deftransform fit(data, opts) do
    if Nx.rank(data) != 2 do
      raise ArgumentError,
            "expected input tensor to have shape {n_samples, n_features},
             got tensor with shape: #{inspect(Nx.shape(data))}"
    end

    opts = NimbleOptions.validate!(opts, @opts_schema)

    metric =
      case opts[:metric] do
        {:minkowski, p} ->
          &Scholar.Metrics.Distance.minkowski(&1, &2, p: p)

        :cosine ->
          &Scholar.Metrics.Distance.cosine/2

        fun when is_function(fun, 2) ->
          fun
      end

    %__MODULE__{
      num_neighbors: opts[:num_neighbors],
      metric: metric,
      data: data,
      batch_size: opts[:batch_size]
    }
  end

  @doc """
  ...

  ## Examples

      iex> data = Nx.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
      iex> model = Scholar.Neighbors.BruteKNN.fit(data, num_neighbors: 2)
      iex> query = Nx.tensor([[1, 3], [4, 2], [3, 6]])
      iex> {neighbors, distances} = Scholar.Neighbors.BruteKNN.predict(model, query)
      iex> neighbors
      #Nx.Tensor<
        u64[3][2]
        [
          [0, 1],
          [1, 2],
          [3, 2]
        ]
      >
      iex> distances
      #Nx.Tensor<
        f32[3][2]
        [
          [1.0, 1.0],
          [2.2360680103302, 2.2360680103302],
          [1.4142135381698608, 2.0]
        ]
      >
  """
  deftransform predict(%__MODULE__{} = model, query) do
    if Nx.rank(query) != 2 do
      raise ArgumentError,
            "expected query tensor to have shape {?, n_features},
             got tensor with shape: #{inspect(Nx.shape(query))}"
    end

    if Nx.axis_size(model.data, 1) != Nx.axis_size(query, 1) do
      raise ArgumentError,
            """
            expected query tensor to have the same dimension as tensor used for fitting the model, \
            got #{inspect(Nx.axis_size(model.data, 1))} \
            and #{inspect(Nx.axis_size(query, 1))}
            """
    end

    predict_n(model, query)
  end

  defn predict_n(%__MODULE__{} = model, query) do
    k = model.num_neighbors
    metric = model.metric
    data = model.data
    type = Nx.Type.merge(to_float_type(data), to_float_type(query))
    query_size = Nx.axis_size(query, 0)

    batch_size =
      case model.batch_size do
        nil -> query_size
        _ -> min(model.batch_size, query_size)
      end

    {batches, leftover} = get_batches(query, batch_size: batch_size)
    num_batches = Nx.axis_size(batches, 0)

    {neighbor_indices, neighbor_distances, _} =
      while {
              neighbor_indices = Nx.broadcast(Nx.u64(0), {query_size, k}),
              neighbor_distances =
                Nx.broadcast(Nx.as_type(:nan, type), {query_size, k}),
              {
                data,
                batches,
                i = Nx.u64(0)
              }
            },
            i < num_batches do
        batch = batches[i]

        {batch_indices, batch_distances} =
          brute_force_search(data, batch, num_neighbors: k, metric: metric)

        # brute_force_search(data, batch, num_neighbors: k)

        neighbor_indices = Nx.put_slice(neighbor_indices, [i * batch_size, 0], batch_indices)

        neighbor_distances =
          Nx.put_slice(neighbor_distances, [i * batch_size, 0], batch_distances)

        {neighbor_indices, neighbor_distances, {data, batches, i + 1}}
      end

    {neighbor_indices, neighbor_distances} =
      case leftover do
        nil ->
          {neighbor_indices, neighbor_distances}

        _ ->
          leftover_size = Nx.axis_size(leftover, 0)

          leftover =
            Nx.slice_along_axis(query, query_size - leftover_size, leftover_size, axis: 0)

          {leftover_indices, leftover_distances} =
            brute_force_search(data, leftover, num_neighbors: k, metric: metric)

          # brute_force_search(data, leftover, num_neighbors: k)

          neighbor_indices =
            Nx.put_slice(neighbor_indices, [num_batches * batch_size, 0], leftover_indices)

          neighbor_distances =
            Nx.put_slice(neighbor_distances, [num_batches * batch_size, 0], leftover_distances)

          {neighbor_indices, neighbor_distances}
      end

    {neighbor_indices, neighbor_distances}
  end

  defnp brute_force_search(data, query, opts) do
    k = opts[:num_neighbors]
    metric = opts[:metric]
    {m, d} = Nx.shape(data)
    n = Nx.axis_size(query, 0)
    x = query |> Nx.new_axis(1) |> Nx.broadcast({n, m, d}) |> Nx.vectorize([:query, :data])
    y = data |> Nx.new_axis(0) |> Nx.broadcast({n, m, d}) |> Nx.vectorize([:query, :data])
    distances = metric.(x, y) |> Nx.devectorize()
    # distances = Scholar.Metrics.Distance.minkowski(x, y) |> Nx.devectorize()

    neighbor_indices =
      Nx.argsort(distances, axis: 1, type: :u64) |> Nx.slice_along_axis(0, k, axis: 1)

    neighbor_distances = Nx.take_along_axis(distances, neighbor_indices, axis: 1)
    {neighbor_indices, neighbor_distances}
  end
end
