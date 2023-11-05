defmodule Scholar.Neighbors.KDTree do
  @moduledoc """
  Implements a kd-tree, a space-partitioning data structure for organizing points
  in a k-dimensional space.

  This is implemented as one-dimensional tensor with indices pointed to highest
  dimension of the given tensor. Traversal starts by calling `root/0` and then
  accessing the `left_child/1` and `right_child/1`. The tree is left-balanced.

  Two construction modes are available:

    * `banded/2` - the tensor has min and max values with an amplitude given by `max - min`.
      It is also guaranteed that the `amplitude * levels(tensor) + 1` does not overflow
      the tensor. See `amplitude/1` to verify if this holds. This implementation happens
      fully within `defn`.

    * `unbanded/2` - there are no known bands (min and max values) to the tensor.
      This implementation is recursive and goes in and out of the `defn`, therefore
      it cannot be called inside `defn`.

  ## References

    * [GPU-friendly, Parallel, and (Almost-)In-Place Construction of Left-Balanced k-d Trees](https://arxiv.org/pdf/2211.00120.pdf).
  """

  import Nx.Defn

  # TODO: Benchmark
  # TODO: Add tagged/amplitude version

  @derive {Nx.Container, keep: [:levels], containers: [:indexes]}
  @enforce_keys [:levels, :indexes]
  defstruct [:levels, :indexes]

  @doc """
  Builds a KDTree without known min-max bounds.

  If your tensor has a known bound (for example, -1 and 1),
  consider using the `banded/2` version which is more efficient.

  ## Options

    * `:compiler` - the default compiler to use for internal defn operations

  ## Examples

      iex> Scholar.Neighbors.KDTree.unbanded(Nx.iota({5, 2}), compiler: EXLA.Defn)
      %Scholar.Neighbors.KDTree{
        levels: 3,
        indexes: Nx.u32([3, 1, 4, 0, 2])
      }

  """
  def unbanded(tensor, opts \\ []) do
    levels = levels(tensor)
    {size, _dims} = Nx.shape(tensor)

    indexes =
      if size > 2 do
        subtree_size = unbanded_subtree_size(1, levels, size)
        {left, mid, right} = Nx.Defn.jit_apply(&root_slice(&1, subtree_size), [tensor], opts)

        acc = <<Nx.to_number(mid)::32-unsigned-native-integer>>
        acc = recur([{1, left}, {2, right}], [], acc, tensor, 1, levels, opts)
        Nx.from_binary(acc, :u32)
      else
        Nx.argsort(tensor[[.., 0]], direction: :desc, type: :u32)
      end

    %__MODULE__{levels: levels, indexes: indexes}
  end

  defp recur([{_i, %Nx.Tensor{shape: {1}} = leaf} | rest], next, acc, tensor, level, levels, opts) do
    [leaf] = Nx.to_flat_list(leaf)
    acc = <<acc::binary, leaf::32-unsigned-native-integer>>
    recur(rest, next, acc, tensor, level, levels, opts)
  end

  defp recur([{i, %Nx.Tensor{shape: {2}} = node} | rest], next, acc, tensor, level, levels, opts) do
    acc = <<acc::binary, Nx.to_number(node[1])::32-unsigned-native-integer>>
    next = [{left_child(i), Nx.slice(node, [0], [1])} | next]
    recur(rest, next, acc, tensor, level, levels, opts)
  end

  defp recur([{i, indexes} | rest], next, acc, tensor, level, levels, opts) do
    %Nx.Tensor{shape: {size, dims}} = tensor
    k = rem(level, dims)
    subtree_size = unbanded_subtree_size(left_child(i), levels, size)

    {left, mid, right} =
      Nx.Defn.jit_apply(&recur_slice(&1, &2, &3, subtree_size), [tensor, indexes, k], opts)

    next = [{right_child(i), right}, {left_child(i), left} | next]
    acc = <<acc::binary, Nx.to_number(mid)::32-unsigned-native-integer>>
    recur(rest, next, acc, tensor, level, levels, opts)
  end

  defp recur([], [], acc, _tensor, _level, _levels, _opts) do
    acc
  end

  defp recur([], next, acc, tensor, level, levels, opts) do
    recur(Enum.reverse(next), [], acc, tensor, level + 1, levels, opts)
  end

  defp root_slice(tensor, subtree_size) do
    indexes = Nx.argsort(tensor[[.., 0]], type: :u32)

    {Nx.slice(indexes, [0], [subtree_size]), indexes[subtree_size],
     Nx.slice(indexes, [subtree_size + 1], [Nx.size(indexes) - subtree_size - 1])}
  end

  defp recur_slice(tensor, indexes, k, subtree_size) do
    sorted = Nx.argsort(Nx.take(tensor, indexes)[[.., k]], type: :u32)
    indexes = Nx.take(indexes, sorted)

    {Nx.slice(indexes, [0], [subtree_size]), indexes[subtree_size],
     Nx.slice(indexes, [subtree_size + 1], [Nx.size(indexes) - subtree_size - 1])}
  end

  defp unbanded_subtree_size(i, levels, size) do
    import Bitwise
    diff = levels - unbanded_level(i) - 1
    shifted = 1 <<< diff
    fllc_s = (i <<< diff) + shifted - 1
    shifted - 1 + min(max(0, size - fllc_s), shifted)
  end

  defp unbanded_level(i) when is_integer(i), do: 31 - clz32(i + 1)

  @doc """
  BANDED
  """
  defn banded(tensor, amplitude) do
    levels = levels(tensor)
    {size, dims} = Nx.shape(tensor)
    band = amplitude + 1
    tags = Nx.broadcast(Nx.u32(0), {size})

    {level, tags, _tensor, _band} =
      while {level = Nx.u32(0), tags, tensor, band}, level < levels - 1 do
        k = rem(level, dims)
        indexes = Nx.argsort(tensor[[.., k]] + band * tags, type: :u32)
        tags = update_tags(tags, indexes, level, levels, size)
        {level + 1, tags, tensor, band}
      end

    k = rem(level, dims)
    indexes = Nx.argsort(tensor[[.., k]] + band * tags, type: :u32)
    %__MODULE__{levels: levels, indexes: indexes}
  end

  defnp update_tags(tags, indexes, level, levels, size) do
    pos = Nx.argsort(indexes, type: :u32)

    pivot =
      banded_segment_begin(tags, levels, size) +
        banded_subtree_size(left_child(tags), levels, size)

    Nx.select(
      pos < (1 <<< level) - 1,
      tags,
      Nx.select(
        pos < pivot,
        left_child(tags),
        Nx.select(
          pos > pivot,
          right_child(tags),
          tags
        )
      )
    )
  end

  defnp banded_subtree_size(i, levels, size) do
    diff = levels - banded_level(i) - 1
    shifted = 1 <<< diff
    first_lowest_level = (i <<< diff) + shifted - 1
    # Use select instead of max to deal with overflows
    lowest_level = Nx.select(first_lowest_level > size, Nx.u32(0), size - first_lowest_level)
    shifted - 1 + min(lowest_level, shifted)
  end

  defn banded_segment_begin(t, levels, size) do
    while t, j <- 0..(size - 1) do
      s = t[j]
      i = (1 <<< banded_level(s)) - 1

      {_, _, acc} =
        while {i, s, acc = i}, i + 1 <= s do
          {i + 1, s, acc + banded_subtree_size(i, levels, size)}
        end

      Nx.put_slice(t, [j], Nx.stack(acc))
    end
  end

  # defn banded_segment_begin(i, levels, size) do
  #   level = banded_level(i)
  #   top = (1 <<< level) - 1
  #   diff = levels - level - 1
  #   shifted = 1 <<< diff
  #   left_siblings = i - top

  #   top + left_siblings * (shifted - 1) +
  #     min(left_siblings * shifted, size - (1 <<< (levels - 1)) - 1)
  # end

  # Since this property relies on u32, let's check the tensor type.
  deftransformp banded_level(%Nx.Tensor{type: {:u, 32}} = i) do
    Nx.subtract(31, Nx.count_leading_zeros(Nx.add(i, 1)))
  end

  @doc """
  Returns the amplitude of a tensor for banding.

  If -1 is returned, it means the tensor cannot use the `banded` algorithm
  to generate a KDTree and `unbanded/2` must be used instead.

  This cannot be invoked inside a `defn`.

  ## Examples

      iex> Scholar.Neighbors.KDTree.amplitude(Nx.iota({10, 2}))
      19
      iex> Scholar.Neighbors.KDTree.amplitude(Nx.iota({20, 2}, type: :f32))
      39.0
      iex> Scholar.Neighbors.KDTree.amplitude(Nx.iota({20, 2}, type: :u8))
      -1

  """
  def amplitude(tensor) do
    max = tensor |> Nx.reduce_max() |> Nx.to_number()
    min = tensor |> Nx.reduce_min() |> Nx.to_number()
    amplitude = max - min
    limit = tensor.type |> Nx.Constants.max_finite() |> Nx.to_number()

    if max + (amplitude + 1) * (Nx.axis_size(tensor, 0) - 1) > limit do
      -1
    else
      amplitude
    end
  end

  @doc """
  Returns the number of resulting levels in a KDTree for `tensor`.

  ## Examples

      iex> Scholar.Neighbors.KDTree.levels(Nx.iota({10, 3}))
      4
  """
  deftransform levels(%Nx.Tensor{} = tensor) do
    case Nx.shape(tensor) do
      {size, _dims} -> 32 - clz32(size)
      _ -> raise ArgumentError, "KDTrees requires a tensor of rank 2"
    end
  end

  @doc """
  Returns the root index.

  ## Examples

      iex> Scholar.Neighbors.KDTree.root()
      0

  """
  deftransform root, do: 0

  @doc """
  Returns the index of the left child of i.

  ## Examples

      iex> Scholar.Neighbors.KDTree.left_child(0)
      1
      iex> Scholar.Neighbors.KDTree.left_child(1)
      3

      iex> Scholar.Neighbors.KDTree.left_child(Nx.u32(3))
      #Nx.Tensor<
        u32
        7
      >

  """
  deftransform left_child(i) when is_integer(i), do: 2 * i + 1
  deftransform left_child(%Nx.Tensor{} = t), do: Nx.add(Nx.multiply(2, t), 1)

  @doc """
  Returns the index of the right child of i.

  ## Examples

      iex> Scholar.Neighbors.KDTree.right_child(0)
      2
      iex> Scholar.Neighbors.KDTree.right_child(1)
      4

      iex> Scholar.Neighbors.KDTree.right_child(Nx.u32(3))
      #Nx.Tensor<
        u32
        8
      >

  """
  deftransform right_child(i) when is_integer(i), do: 2 * i + 2
  deftransform right_child(%Nx.Tensor{} = t), do: Nx.add(Nx.multiply(2, t), 2)

  @clz_lookup {32, 31, 30, 30, 29, 29, 29, 29, 28, 28, 28, 28, 28, 28, 28, 28}

  defp clz32(x) when is_integer(x) do
    import Bitwise

    n =
      if x >= 1 <<< 16 do
        if x >= 1 <<< 24 do
          if x >= 1 <<< 28, do: 28, else: 24
        else
          if x >= 1 <<< 20, do: 20, else: 16
        end
      else
        if x >= 1 <<< 8 do
          if x >= 1 <<< 12, do: 12, else: 8
        else
          if x >= 1 <<< 4, do: 4, else: 0
        end
      end

    elem(@clz_lookup, x >>> n) - n
  end
end
