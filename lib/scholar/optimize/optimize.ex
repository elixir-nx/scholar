defmodule Scholar.Optimize do
  @moduledoc """
  Optimization routines for minimizing scalar functions.

  This module provides general-purpose optimization functionality similar to
  SciPy's `scipy.optimize`.

  ## Scalar (Univariate) Optimization

  Use `minimize_scalar/2` for functions of a single variable:

      result = Scholar.Optimize.minimize_scalar(
        fn x -> (x - 2) ** 2 end,
        bracket: {0.0, 4.0}
      )

  ## Available Methods

  ### Scalar
  * `:golden` - Golden section search
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:x, :fun, :converged, :iterations, :fun_evals, :grad_evals]}
  defstruct [:x, :fun, :converged, :iterations, :fun_evals, :grad_evals]

  @type t :: %__MODULE__{
          x: Nx.Tensor.t(),
          fun: Nx.Tensor.t(),
          converged: Nx.Tensor.t(),
          iterations: Nx.Tensor.t(),
          fun_evals: Nx.Tensor.t(),
          grad_evals: Nx.Tensor.t() | nil
        }

  minimize_scalar_opts = [
    method: [
      type: {:in, [:golden]},
      default: :golden,
      doc: """
      Optimization method to use:
      * `:golden` - Golden section search (default)
      """
    ],
    bracket: [
      type: {:custom, Scholar.Options, :bracket, []},
      required: true,
      doc: """
      Bracket containing the minimum. A tuple `{a, b}` defining the search interval.
      The minimum must lie within this interval.
      """
    ],
    tol: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-8,
      doc: "Absolute tolerance for convergence."
    ],
    maxiter: [
      type: :pos_integer,
      default: 500,
      doc: "Maximum number of iterations."
    ]
  ]

  @minimize_scalar_schema NimbleOptions.new!(minimize_scalar_opts)

  @doc """
  Minimize a scalar function of one variable.

  ## Arguments

  * `fun` - The objective function to minimize. Must be a defn-compatible function
    that takes a scalar tensor and returns a scalar tensor.
  * `opts` - Options (see below).

  ## Options

  #{NimbleOptions.docs(@minimize_scalar_schema)}

  ## Returns

  A `Scholar.Optimize` struct where `:x` is a scalar tensor.

  ## Examples

      iex> fun = fn x -> (x - 3) ** 2 end
      iex> result = Scholar.Optimize.minimize_scalar(fun, bracket: {0.0, 5.0})
      iex> Nx.to_number(result.converged)
      1
      iex> Nx.all_close(result.x, Nx.tensor(3.0), atol: 1.0e-4) |> Nx.to_number()
      1
  """
  deftransform minimize_scalar(fun, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @minimize_scalar_schema)

    case opts[:method] do
      :golden ->
        Scholar.Optimize.GoldenSection.minimize(fun, opts)
    end
  end
end
