# Best Practices for Scholar.Optimize Contributions

This document captures key patterns and requirements for implementing optimization algorithms in Scholar, based on review feedback from José Valim (PR #323, #327).

## Core Principle: JIT/GPU Compatibility

**All optimization algorithms must be JIT-compilable and GPU-compatible.**

This means the main entry point must be a `defn` function that can be called with `Nx.Defn.jit_apply/3`.

## Required Patterns

### 1. Use `defn` for Entry Points

```elixir
# GOOD - JIT compatible
defn minimize(a, b, fun, opts \\ []) do
  {tol, maxiter} = transform_opts(opts)
  minimize_n(a, b, fun, tol, maxiter)
end

# BAD - NOT JIT compatible
deftransform minimize(fun, opts) do
  # This prevents JIT compilation!
end
```

### 2. Expose Required Parameters as Function Arguments

Don't bury required parameters in options - expose them as explicit arguments:

```elixir
# GOOD - bounds as explicit args
defn minimize(a, b, fun, opts \\ [])

# BAD - bounds buried in options
deftransform minimize(fun, opts) do
  {a, b} = opts[:bracket]
```

**Why?** The `deftransform -> defn` conversion will check the input types automatically, eliminating the need for custom validation logic.

### 3. Use `deftransformp` Only for Option Validation

```elixir
deftransformp transform_opts(opts) do
  opts = NimbleOptions.validate!(opts, @opts_schema)
  {opts[:tol], opts[:maxiter]}
end
```

### 4. Use `Nx.select` for Branch-Free Conditionals

Never use Elixir runtime conditionals (`if`, `cond`, `case`) inside `defn` functions:

```elixir
# GOOD - tensor-based conditional
new_a = Nx.select(condition, value_if_true, value_if_false)

# BAD - Elixir runtime conditional
new_a = if Nx.to_number(condition) == 1, do: value_if_true, else: value_if_false
```

For complex multi-way conditionals, use nested `Nx.select`:

```elixir
# Four-way conditional
result = Nx.select(
  cond1,
  value1,
  Nx.select(
    cond2,
    value2,
    Nx.select(cond3, value3, value4)
  )
)
```

### 5. Use `while` Loop (Not Recursion)

```elixir
# GOOD - while loop
{final_state, _} =
  while {state = initial_state, {tol, maxiter}},
        state.iter < maxiter and state.b - state.a >= tol do
    # Update state
    new_state = %{state | iter: state.iter + 1, ...}
    {new_state, {tol, maxiter}}
  end

# BAD - recursive call (not JIT-compatible)
defp loop(fun, state, tol, maxiter) do
  if converged?(state) do
    state
  else
    loop(fun, update(state), tol, maxiter)
  end
end
```

### 6. Never Use `Nx.to_number` in `defn`

All computation must stay as tensors for JIT compilation:

```elixir
# GOOD - all tensor operations
converged = state.b - state.a < tol

# BAD - converts to Elixir number
converged = Nx.to_number(state.b) - Nx.to_number(state.a) < Nx.to_number(tol)
```

### 7. Use Unsigned Types for Non-Negative Counters

```elixir
initial_state = %{
  iter: Nx.u32(0),      # u32 for iteration count
  f_evals: Nx.u32(2)    # u32 for function evaluation count
}
```

### 8. Let Users Control Precision via Input Types

Don't force type conversions - let the tensor type propagate from inputs:

```elixir
# GOOD - let user decide precision
defn minimize(a, b, fun, opts \\ []) do
  # a and b types propagate through computation
end

# BAD - forcing f64
a = Nx.tensor(a, type: :f64)
```

Document that users can use f64 tensors for higher precision:

```elixir
@doc """
For higher precision, use f64 tensors for bounds:

    a = Nx.tensor(0.0, type: :f64)
    b = Nx.tensor(5.0, type: :f64)
    result = Brent.minimize(a, b, fun, tol: 1.0e-10)
"""
```

### 9. Use Module Constants Directly

```elixir
# GOOD - use @attr directly in defn
@phi 0.6180339887498949

defnp minimize_n(a, b, fun, tol, maxiter) do
  c = b - @phi * (b - a)
end

# BAD - wrapping in tensor
defnp minimize_n(a, b, fun, tol, maxiter) do
  phi = Nx.tensor(@phi)
  c = b - phi * (b - a)
end
```

### 10. Self-Contain Modules

Keep NimbleOptions validation in the same module - don't create wrapper modules:

```elixir
defmodule Scholar.Optimize.Brent do
  opts = [
    tol: [...],
    maxiter: [...]
  ]

  @opts_schema NimbleOptions.new!(opts)

  # Validation happens here, not in a separate module
end
```

## Module Structure Template

```elixir
defmodule Scholar.Optimize.AlgorithmName do
  @moduledoc """
  Description of the algorithm.

  ## Algorithm
  ...

  ## Convergence
  ...

  ## References
  ...
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:x, :fun, :converged, :iterations, :fun_evals]}
  defstruct [:x, :fun, :converged, :iterations, :fun_evals]

  @type t :: %__MODULE__{
          x: Nx.Tensor.t(),
          fun: Nx.Tensor.t(),
          converged: Nx.Tensor.t(),
          iterations: Nx.Tensor.t(),
          fun_evals: Nx.Tensor.t()
        }

  # Constants
  @some_constant 0.123456789

  # Options schema
  opts = [
    tol: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-5,
      doc: "..."
    ],
    maxiter: [
      type: :pos_integer,
      default: 500,
      doc: "..."
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Main entry point documentation...
  """
  defn minimize(a, b, fun, opts \\ []) do
    {tol, maxiter} = transform_opts(opts)
    minimize_n(a, b, fun, tol, maxiter)
  end

  deftransformp transform_opts(opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    {opts[:tol], opts[:maxiter]}
  end

  defnp minimize_n(a, b, fun, tol, maxiter) do
    # Implementation using while loop and Nx.select
  end
end
```

## Test Requirements

Every optimization module must include:

1. **Basic functionality tests** - Verify correct results on standard test functions
2. **Option handling tests** - Test tolerance and maxiter options
3. **JIT compatibility test** - Critical! Must pass:

```elixir
test "works with jit_apply" do
  fun = fn x -> Nx.pow(Nx.subtract(x, 3), 2) end
  opts = [tol: 1.0e-5, maxiter: 500]

  result = Nx.Defn.jit_apply(&AlgorithmName.minimize/4, [0.0, 5.0, fun, opts])

  assert Nx.to_number(result.converged) == 1
end
```

4. **Tensor bounds test** - Accept both numbers and tensors
5. **Precision test** - Higher precision with f64 bounds

## Validation Against SciPy

When implementing algorithms, validate results against SciPy:

```python
from scipy.optimize import minimize_scalar

result = minimize_scalar(func, bracket=(a, b), method='brent')
print(f"x: {result.x}, f(x): {result.fun}, iterations: {result.nit}")
```

Use these reference values in tests with appropriate tolerance (typically `atol: 1.0e-4` to `1.0e-6`).

## References

- PR #323: https://github.com/elixir-nx/scholar/pull/323 (original comprehensive optimizer)
- PR #327: https://github.com/elixir-nx/scholar/pull/327 (merged Golden Section)
- SciPy optimize: https://docs.scipy.org/doc/scipy/tutorial/optimize.html
