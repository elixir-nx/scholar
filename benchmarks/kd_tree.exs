# mix run benchmarks/kd_tree.exs
Nx.global_default_backend(EXLA.Backend)
Nx.Defn.global_default_options(compiler: EXLA)

key = Nx.Random.key(System.os_time())
{uniform, _new_key} = Nx.Random.uniform(key, shape: {1000, 3})

Benchee.run(
  %{
    "unbounded" => fn -> Scholar.Neighbors.KDTree.unbounded(uniform) end,
    "bounded" => fn -> Scholar.Neighbors.KDTree.bounded(uniform, 2) end
  },
  time: 10,
  memory_time: 2
)
