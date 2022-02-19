defmodule Scholar.MixProject do
  use Mix.Project

  def project do
    [
      app: :scholar,
      version: "0.1.0",
      elixir: "~> 1.14-dev",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.1.0"}
    ]
  end
end
