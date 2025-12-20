<p align="center">
  <img src="https://github.com/elixir-nx/scholar/raw/main/images/scholar.png" alt="Scholar" width="400">
  <br />
  <a href="https://hexdocs.pm/scholar"><img src="http://img.shields.io/badge/hex.pm-docs-green.svg?style=flat" title="Documentation" /></a>
  <a href="https://hex.pm/packages/scholar"><img src="https://img.shields.io/hexpm/v/scholar.svg" title="Package" /></a>
</p>

<br />

Traditional machine learning tools built on top of Nx. Scholar implements
several algorithms for classification, regression, clustering, dimensionality
reduction, metrics, and preprocessing. For deep learning, see
[Axon](https://github.com/elixir-nx/axon).

## Installation

### Mix projects

Add to your `mix.exs`:

```elixir
def deps do
  [
    {:scholar, "~> 0.3.0"}
  ]
end
```

Besides Scholar, you will most likely want to use an existing Nx compiler/backend,
such as EXLA:

```elixir
def deps do
  [
    {:scholar, "~> 0.3.0"},
    {:exla, ">= 0.0.0"}
  ]
end
```

And then in your `config/config.exs` file:

```elixir
import Config
config :nx, :default_backend, EXLA.Backend
# Client can also be set to :cuda / :rocm
config :nx, :default_defn_options, [compiler: EXLA, client: :host]
```

> #### JIT required! {: .warning}
>
> It is important you set the `default_defn_options` as shown in the snippet above,
> as many algorithms in Scholar use loops which are much more memory efficient when
> JIT compiled.
>
> If for some reason you cannot set a default `defn` compiler, you can explicitly
> JIT any function, for example: `EXLA.jit(&Scholar.Cluster.AffinityPropagation.fit/1)`.

### Notebooks

To use Scholar inside code notebooks, run:

```elixir
Mix.install([
  {:scholar, "~> 0.3.0"},
  {:exla, ">= 0.0.0"}
])

Nx.global_default_backend(EXLA.Backend)
# Client can also be set to :cuda / :rocm
Nx.Defn.global_default_options(compiler: EXLA, client: :host)
```

> #### JIT required! {: .warning}
>
> It is important you set the `Nx.Defn.global_default_options/1` as shown in the snippet
> above, as many algorithms in Scholar use loops which are much more memory efficient
> when JIT compiled.
>
> If for some reason you cannot set a default `defn` compiler, you can explicitly
> JIT any function, for example: `EXLA.jit(&Scholar.Cluster.AffinityPropagation.fit/1)`.

## Contributing

We welcome the contribution of new algorithms to the project. However, it is important
to note that we only accept implementations that are fully implemented as "numerical
definitions", as this gives us the ability to compile and run all algorithms inside
GPUs. This means not all algorithms can be implemented in Scholar. Decision
trees/forests are one of such algorithms and for those there are additional libraries,
such as [EXGBoost](https://github.com/acalejos/exgboost).

Implementation wise, this means most functions simply validate options and
then delegate to an implementation fully written inside a `defn` or `defnp`.
You can look [at this pull request as an example](https://github.com/elixir-nx/scholar/pull/314).
We also recommend adding tests that show that you can still invoke your
imlpementation after wrapping it in a `Nx.Defn.jit/2` call.

## License

Copyright (c) 2022 The Machine Learning Working Group of the Erlang Ecosystem Foundation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
