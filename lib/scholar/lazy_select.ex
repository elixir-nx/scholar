defmodule Scholar.LazySelect do
  import Nx.Defn

  deftransform lazy_select(tensor, opts \\ []) do
    k = opts[:k]

    if Nx.rank(tensor) != 1 do
      raise ArgumentError, "expected a vector, got a #{inspect(tensor)}"
    end

    if k >= Nx.size(tensor) do
      raise ArgumentError, "k must be less than the size of the tensor"
    end

    len = Nx.size(tensor)
    selection_len = :math.pow(len, 3 / 4) |> trunc()
    key = Nx.Random.key(System.system_time())
    opts = Keyword.put(opts, :selection_len, selection_len)
    lazy_select_core(tensor, key, opts)
  end

  defnp lazy_select_core(tensor, key, opts \\ []) do
    k = opts[:k]
    selection_len = opts[:selection_len]

    type = Nx.type(tensor)
    max_val = Nx.Constants.max(type)

    len = Nx.size(tensor)
    x = Nx.as_type(k * Nx.pow(len, -1 / 4), :s64)
    len_sqrt = Nx.sqrt(len)
    low_index = Nx.as_type(Nx.max(0, x - len_sqrt), :s64)
    high_index = Nx.as_type(Nx.min(selection_len - 1, x + len_sqrt), :s64)

    termination_condition = Nx.u8(0)

    {result, _} =
      while {res = Nx.tensor(0, type: Nx.type(tensor)),
             {tensor, key, max_val, termination_condition}},
            termination_condition != 1 do
        {candidates, new_key} = Nx.Random.choice(key, tensor, samples: selection_len)
        candidates = Nx.sort(candidates)

        low = Nx.take(candidates, low_index)
        high = Nx.take(candidates, high_index)

        low_pos = Nx.sum(low > tensor)
        high_pos = Nx.sum(high > tensor)

        indices_to_append = tensor >= low and tensor <= high

        {agg_index, agg, _} =
          while {curr_agg_index = Nx.tensor([0]),
                 agg = Nx.broadcast(max_val, {Kernel.min(4 * selection_len + 2, len)}),
                 {tensor, indices_to_append, i = 0}},
                i < len and Nx.reshape(curr_agg_index != 4 * selection_len + 2, {}) do
            if indices_to_append[i] == Nx.u8(1) do
              {curr_agg_index + 1, Nx.indexed_put(agg, curr_agg_index, tensor[i]),
               {tensor, indices_to_append, i + 1}}
            else
              {curr_agg_index, agg, {tensor, indices_to_append, i + 1}}
            end
          end

        termination_condition =
          Nx.reshape(low_pos <= k and k <= high_pos and agg_index <= 4 * selection_len + 1, {})

        res =
          if termination_condition do
            Nx.sort(agg[0..Kernel.min(4 * selection_len, len - 1)])[k - low_pos]
          else
            res
          end

        {res, {tensor, new_key, max_val, termination_condition}}
      end

    result
  end
end
