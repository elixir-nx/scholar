<h1><img src="./images/scholar.png" alt="Scholar" width="400"></h1>

[![Documentation](http://img.shields.io/badge/hex.pm-docs-green.svg?style=flat)](https://hexdocs.pm/scholar)
[![Package](https://img.shields.io/hexpm/v/scholar.svg)](https://hex.pm/packages/scholar)

Traditional machine learning tools built on top of Nx. Scholar implements
several algorithms for classification, regression, clustering, dimensionality
reduction, metrics, and preprocessing.

For deep learning, see [Axon](https://github.com/elixir-nx/axon).

## Installation

### Mix projects

Add to your `mix.exs`:

```elixir
def deps do
  [
    {:scholar, "~> 0.1"}
  ]
end
```

Besides Scholar, you will most likely want to use an existing Nx compiler/backend,
such as EXLA:

```elixir
def deps do
  [
    {:scholar, "~> 0.1"},
    {:exla, ">= 0.0.0"}
  ]
end
```

And then in your `config/config.exs` file:

```elixir
import Config
config :nx, :default_backend, EXLA.Backend
```

### Notebooks

To use Scholar inside code notebooks, run:

```elixir
Mix.install([
  {:scholar, "~> 0.1"},
  {:exla, ">= 0.0.0"}
])

Nx.global_default_backend(EXLA.Backend)
```

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
