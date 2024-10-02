defmodule Scholar.Decomposition.Utils do
  @moduledoc false
  import Nx.Defn
  require Nx

  defn flip_svd(u, v) do
    max_abs_cols_idx = u |> Nx.abs() |> Nx.argmax(axis: 0, keep_axis: true)
    signs = u |> Nx.take_along_axis(max_abs_cols_idx, axis: 0) |> Nx.sign() |> Nx.squeeze()
    u = u * signs
    v = v * Nx.new_axis(signs, -1)
    {u, v}
  end
end
