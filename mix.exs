defmodule Scholar.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/scholar"
  @version "0.2.1"

  def project do
    [
      app: :scholar,
      name: "Scholar",
      version: @version,
      elixir: "~> 1.14",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      package: package()
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:ex_doc, "~> 0.30", only: :docs},
      # {:nx, "~> 0.6", override: true},
      {:nx, github: "elixir-nx/nx", sparse: "nx", override: true, branch: "v0.6"},
      {:nimble_options, "~> 0.5.2 or ~> 1.0"},
      {:exla, "~> 0.6", optional: true},
      {:polaris, "~> 0.1"},
      {:benchee, "~> 1.0", only: :dev}
    ]
  end

  defp package do
    [
      maintainers: ["Mateusz Słuszniak"],
      description: "Traditional machine learning on top of Nx",
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp docs do
    [
      main: "readme",
      source_url: @source_url,
      logo: "images/scholar_simplified.png",
      extra_section: "Guides",
      extras: [
        "README.md",
        "notebooks/linear_regression.livemd",
        "notebooks/k_means.livemd",
        "notebooks/k_nearest_neighbors.livemd",
        "notebooks/cv_gradient_boosting_tree.livemd"
      ],
      groups_for_modules: [
        Models: [
          Scholar.Cluster.AffinityPropagation,
          Scholar.Cluster.DBSCAN,
          Scholar.Cluster.GaussianMixture,
          Scholar.Cluster.KMeans,
          Scholar.Decomposition.PCA,
          Scholar.Integrate,
          Scholar.Interpolation.BezierSpline,
          Scholar.Interpolation.CubicSpline,
          Scholar.Interpolation.Linear,
          Scholar.Linear.IsotonicRegression,
          Scholar.Linear.LinearRegression,
          Scholar.Linear.LogisticRegression,
          Scholar.Linear.PolynomialRegression,
          Scholar.Linear.RidgeRegression,
          Scholar.Linear.SVM,
          Scholar.Manifold.TSNE,
          Scholar.NaiveBayes.Complement,
          Scholar.NaiveBayes.Gaussian,
          Scholar.NaiveBayes.Multinomial,
          Scholar.Neighbors.KNearestNeighbors,
          Scholar.Neighbors.RadiusNearestNeighbors
        ],
        Utilities: [
          Scholar.Covariance,
          Scholar.Impute.SimpleImputer,
          Scholar.Metrics.Classification,
          Scholar.Metrics.Clustering,
          Scholar.Metrics.Distance,
          Scholar.Metrics.Regression,
          Scholar.Metrics.Similarity,
          Scholar.ModelSelection,
          Scholar.Preprocessing,
          Scholar.Stats
        ]
      ],
      before_closing_body_tag: &before_closing_body_tag/1
    ]
  end

  defp before_closing_body_tag(:html) do
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.css" integrity="sha384-t5CR+zwDAROtph0PXGte6ia8heboACF9R5l/DiY+WZ3P2lxNgvJkQk5n7GPvLMYw" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.js" integrity="sha384-FaFLTlohFghEIZkw6VGwmf9ISTubWAVYW8tG8+w2LAIftJEULZABrF9PPFv+tVkH" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/contrib/auto-render.min.js" integrity="sha384-bHBqxz8fokvgoJ/sc17HODNxa42TlaEhB+w8ZJXTc2nZf1VgEaFZeZvT4Mznfz0v" crossorigin="anonymous"></script>
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ]
        });
      });
    </script>
    <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.1"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.18.2"></script>
    <script>
      document.addEventListener("DOMContentLoaded", function () {
        for (const codeEl of document.querySelectorAll("pre code.vega-lite")) {
          try {
            const preEl = codeEl.parentElement;
            const spec = JSON.parse(codeEl.textContent);
            const plotEl = document.createElement("div");
            preEl.insertAdjacentElement("afterend", plotEl);
            vegaEmbed(plotEl, spec);
            preEl.remove();
          } catch (error) {
            console.log("Failed to render Vega-Lite plot: " + error)
          }
        }
      });
    </script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
