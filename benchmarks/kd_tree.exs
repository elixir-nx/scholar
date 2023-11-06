# mix run benchmarks/kd_tree.exs
Nx.global_default_backend(EXLA.Backend)
Nx.Defn.global_default_options(compiler: EXLA)

key = Nx.Random.key(System.os_time())
{uniform, _new_key} = Nx.Random.uniform(key, shape: {1000, 3})

Benchee.run(
  %{
    "unbound" => fn -> Scholar.Neighbors.KDTree.unbound(uniform) end,
    "bound" => fn -> Scholar.Neighbors.KDTree.bound(uniform, 2) end
  },
  time: 10,
  memory_time: 2
)
