defmodule ScholarTest do
  use ExUnit.CaseTemplate

  def assert_all_close(left, right, opts \\ []) do
    atol = opts[:atol] || 1.0e-7
    rtol = opts[:rtol] || 1.0e-7

    equals =
      left
      |> Nx.all_close(right, atol: atol, rtol: rtol)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    if equals != Nx.tensor(1, type: {:u, 8}, backend: Nx.BinaryBackend) do
      flunk("""
      expected
      #{inspect(left)}
      to be within tolerance of
      #{inspect(right)}
      """)
    end
  end
end
