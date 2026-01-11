defmodule Scholar.Decomposition.Utils do
  @moduledoc false
  import Nx.Defn

  defn flip_svd(u, v, u_based \\ true) do
    base =
      if u_based do
        u
      else
        Nx.transpose(v)
      end

    max_abs_cols_idx = base |> Nx.abs() |> Nx.argmax(axis: 0, keep_axis: true)
    signs = base |> Nx.take_along_axis(max_abs_cols_idx, axis: 0) |> Nx.sign() |> Nx.squeeze()
    u = u * signs
    v = v * Nx.new_axis(signs, -1)
    {u, v}
  end
end
