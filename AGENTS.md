# Best Practices for Scholar Contributions

This document captures key patterns and requirements for implementing algorithms in Scholar.

## Core Principle: JIT/GPU Compatibility

**All algorithms must be JIT-compilable and GPU-compatible.**

Nx relies on multi-stage compilation. When `defn` is executed, it doesn't have
actually values, only references to tensors shapes and types. The `defn` execution
then builds a numeric graph, which is lowered just-in-time (JIT) to CPUs and GPUs.

## Required Patterns

### 1. Use `deftransform` for Entry Points

The main entry point must be a `deftransform` that simply unpacks
options and immediately calls a `defnp`.

```elixir
# GOOD - JIT compatible
deftransform fit(a, b, fun, opts \\ []) do
  opts = NimbleOptions.validate!(opts, @opts_schema)
  fit_n(a, b, fun, opts[:tol], opts[:maxiter])
end
```

Do not perform `Nx` operations inside `deftransform` (not even during validation).
Make sure all values that can be tensors are given as explicit arguments to the
`defnp` function, and invoke `Nx` operations there.

### 2. Expose Required Parameters as Function Arguments

Don't bury required parameters in options - expose them as explicit arguments:

```elixir
# GOOD - bounds as explicit args
deftransform fit(a, b, fun, opts \\ [])

# BAD - bounds buried in options
deftransform fit(fun, opts) do
  {a, b} = opts[:bracket]
```

**Why?** Note the `deftransform -> defn` conversion will convert input types to Nx tensors when crossing the `def -> defn` boundary, eliminating the need for custom validation logic.

### 3. Use `Nx.select` for Branch-Free Conditionals

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

### 4. Use `while` Loop (Not Recursion)

```elixir
# GOOD - while loop
{final_state, _} =
  while {state = initial_state, {tol, maxiter}},
        state.iter < maxiter and state.b - state.a >= tol do
    # Update state
    new_state = %{state | iter: state.iter + 1, ...}
    {new_state, {tol, maxiter}}
  end

# BAD - recursive call
defnp loop(fun, state, tol, maxiter) do
  if converged?(state) do
    state
  else
    loop(fun, update(state), tol, maxiter)
  end
end
```

**Why?** Remember when `defnp` runs, we don't have runtime values,
so most loops can't terminate within `defnp`. And for the loops that can
terminate (the condition is known statically), doing the loop in `defnp`
means that you are generating the graph recursively, which can then become
large and take a long time to compile.

### 5. Never Use `Nx.to_number` in `defn`

All computation must stay as tensors for JIT compilation:

```elixir
# GOOD - all tensor operations
converged = state.b - state.a < tol

# BAD - converts to Elixir number (will crash inside JIT)
converged = Nx.to_number(state.b) - Nx.to_number(state.a) < Nx.to_number(tol)
```

### 6. Use Unsigned Types for Non-Negative Counters

```elixir
initial_state = %{
  iter: Nx.u32(0),      # u32 for iteration count
  f_evals: Nx.u32(2)    # u32 for function evaluation count
}
```

### 7. Let Users Control Precision via Input Types

Don't force type conversions - let the tensor type propagate from inputs:

```elixir
# GOOD - let user decide precision
defn minimize(a, b, fun, opts \\ []) do
  # a and b types propagate through computation
end

# BAD - forcing f64
a = Nx.tensor(a, type: :f64)
```

### 8. Use Module Constants Directly

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

### 9. Self-Contain Modules

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

## Test Requirements

Every module must include:

1. **Basic functionality tests** - Verify correct results on standard test functions
2. **Option handling tests** - Test options
3. **JIT compatibility test** - Critical! For example, if the algorithm has a `minimize` function, ensure it can be invoked when wrapped in `jit_apply/4`:

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

Use these reference values in tests with appropriate tolerance.

## References

- PR #323: https://github.com/elixir-nx/scholar/pull/314 (RobustScaler)
- PR #327: https://github.com/elixir-nx/scholar/pull/327 (GoldenSection)
