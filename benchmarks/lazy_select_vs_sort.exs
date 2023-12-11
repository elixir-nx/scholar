# mix run benchmarks/lazy_select_vs_sort.exs
Nx.global_default_backend(EXLA.Backend)
Nx.Defn.global_default_options(compiler: EXLA)

key = Nx.Random.key(System.os_time())

inputs_knn = %{
  "10" => elem(Nx.Random.shuffle(key, Nx.iota({10})), 0),
  "100" => elem(Nx.Random.shuffle(key, Nx.iota({100})), 0),
  "1000" => elem(Nx.Random.shuffle(key, Nx.iota({1000})), 0),
  "10000" => elem(Nx.Random.shuffle(key, Nx.iota({10000})), 0),
  "100000" => elem(Nx.Random.shuffle(key, Nx.iota({100000})), 0)
}

Benchee.run(
  %{
    "lazy_select" => fn x ->
      EXLA.jit_apply(&Scholar.LazySelect.lazy_select(&1, k: div(Nx.size(&1), 2)), [x])
    end,
    "sort" => fn x ->
      EXLA.jit_apply(&Nx.sort(&1), [x])[div(Nx.size(x), 2)]
    end,
  },
  time: 10,
  memory_time: 2,
  inputs: inputs_knn
)


# 22:55:47.179 [info] TfrtCpuClient created.
# Operating System: Linux
# CPU Information: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
# Number of Available Cores: 6
# Available memory: 15.61 GB
# Elixir 1.15.5
# Erlang 26.1.2

# Benchmark suite executing with the following configuration:
# warmup: 2 s
# time: 10 s
# memory time: 2 s
# reduction time: 0 ns
# parallel: 1
# inputs: 10, 100, 1000, 10000, 100000
# Estimated total run time: 2.33 min

# Benchmarking lazy_select with input 10 ...
# Benchmarking lazy_select with input 100 ...
# Benchmarking lazy_select with input 1000 ...
# Benchmarking lazy_select with input 10000 ...
# Benchmarking lazy_select with input 100000 ...
# Benchmarking sort with input 10 ...
# Benchmarking sort with input 100 ...
# Benchmarking sort with input 1000 ...
# Benchmarking sort with input 10000 ...
# Benchmarking sort with input 100000 ...

# ##### With input 10 #####
# Name                  ips        average  deviation         median         99th %
# lazy_select       61.12 K       16.36 μs   ±164.20%       13.69 μs       55.52 μs
# sort              22.62 K       44.20 μs   ±140.67%       36.76 μs      158.98 μs

# Comparison:
# lazy_select       61.12 K
# sort              22.62 K - 2.70x slower +27.84 μs

# Memory usage statistics:

# Name           Memory usage
# lazy_select         3.84 KB
# sort               15.15 KB - 3.95x memory usage +11.31 KB

# **All measurements for memory usage were the same**

# ##### With input 100 #####
# Name                  ips        average  deviation         median         99th %
# lazy_select       40.76 K       24.54 μs   ±110.57%       21.23 μs       76.69 μs
# sort              19.65 K       50.89 μs   ±145.68%       42.19 μs      167.57 μs

# Comparison:
# lazy_select       40.76 K
# sort              19.65 K - 2.07x slower +26.36 μs

# Memory usage statistics:

# Name           Memory usage
# lazy_select         3.84 KB
# sort               15.15 KB - 3.95x memory usage +11.31 KB

# **All measurements for memory usage were the same**

# ##### With input 1000 #####
# Name                  ips        average  deviation         median         99th %
# sort               6.40 K      156.32 μs    ±95.19%      143.73 μs      251.71 μs
# lazy_select        5.38 K      185.86 μs    ±62.22%      175.89 μs      293.33 μs

# Comparison:
# sort               6.40 K
# lazy_select        5.38 K - 1.19x slower +29.54 μs

# Memory usage statistics:

# Name           Memory usage
# sort               15.15 KB
# lazy_select         3.84 KB - 0.25x memory usage -11.31250 KB

# **All measurements for memory usage were the same**

# ##### With input 10000 #####
# Name                  ips        average  deviation         median         99th %
# sort               624.70        1.60 ms    ±23.14%        1.58 ms        2.02 ms
# lazy_select         86.37       11.58 ms     ±3.98%       11.50 ms       12.96 ms

# Comparison:
# sort               624.70
# lazy_select         86.37 - 7.23x slower +9.98 ms

# Memory usage statistics:

# Name           Memory usage
# sort               15.15 KB
# lazy_select         3.84 KB - 0.25x memory usage -11.31250 KB

# **All measurements for memory usage were the same**

# ##### With input 100000 #####
# Name                  ips        average  deviation         median         99th %
# sort                49.39       20.25 ms     ±4.29%       20.12 ms       24.96 ms
# lazy_select          1.14      878.75 ms     ±3.60%      869.93 ms      938.22 ms

# Comparison:
# sort                49.39
# lazy_select          1.14 - 43.40x slower +858.50 ms

# Memory usage statistics:

# Name           Memory usage
# sort               15.15 KB
# lazy_select         3.84 KB - 0.25x memory usage -11.31250 KB

# **All measurements for memory usage were the same**
