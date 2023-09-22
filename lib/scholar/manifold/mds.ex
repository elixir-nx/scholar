defmodule Scholar.Manifold.MDS do
  @moduledoc """
  TSNE (t-Distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction technique.

  ## References

  * [t-SNE: t-Distributed Stochastic Neighbor Embedding](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
  """
  import Nx.Defn
  import Scholar.Shared
  alias Scholar.Metrics.Distance

  # initialize x randomly or pass the init x earlier
  defnp smacof(dissimilarities, x, max_iter, opts) do
    num_samples = Nx.axis_size(dissimilarities, 0)
    similarities_flat = Nx.flatten((1 - Nx.tri(num_samples)) * dissimilarities)
    similarities_flat_indices = remove_main_diag_indices(similarities_flat)

    n = Nx.axis_size(dissimilarities, 0)

    similarities_flat_w =
      Nx.take(similarities_flat, similarities_flat_indices) |> Nx.reshape({n, n - 1})

    res =
      while {{x, stress = Nx.Constants.infinity(), i = 0}, dissimilarities, max_iter,
             similarities_flat_indices, similarities_flat, old_stress = Nx.Constants.infinity(),
             stop_value = 0},
            i < max_iter and not stop_value do
        dis = Distance.pairwise_euclidean(x)

        disparities =
          if opts[:metric] do
            dissimilarities
          else
            dis_flat = Nx.flatten(dis)

            dis_flat_indices = remove_main_diag_indices(dis_flat)

            n = Nx.axis_size(dis, 0)

            dis_flat_w = Nx.take(dis_flat, dis_flat_indices) |> Nx.reshape({n, n - 1})
            # dis_flat_w = Nx.flatten(remove_main_diag(dis))

            disparities_flat =
              Scholar.Linear.IsotonicRegression.fit(similarities_flat_w, dis_flat_w)

            disparities_flat =
              Scholar.Linear.IsotonicRegression.predict(disparities_flat, similarities_flat_w)

            # disparities = Nx.select(similarities_flat != 0, disparities_flat, disparities)

            disparities = Nx.indexed_put(dis_flat, similarities_flat_indices, disparities_flat)
            disparities = Nx.reshape(disparities, {n, n})

            disparities * Nx.sum(Nx.sqrt(n * (n - 1) / 2 / disparities ** 2))
          end

        stress = Nx.sum((Nx.flatten(dis) - Nx.flatten(disparities)) ** 2) / 2

        stress =
          if opts[:normalized_stress] do
            Nx.sqrt(stress / (Nx.sum(Nx.flatten(disparities) ** 2) / 2))
          else
            stress
          end

        dis = Nx.select(dis == 0, 1.0e-5, dis)
        ratio = disparities / dis
        b = -ratio
        b = Nx.put_diagonal(b, Nx.take_diagonal(b) + Nx.sum(ratio, axes: [1]))
        x = 1.0 / n * Nx.dot(b, x)

        dis = Nx.sum(Nx.sqrt(Nx.sum(x ** 2, axes: [1])))

        stop_value = if old_stress - stress / dis < opts[:eps], do: 1, else: 0

        old_stress = stress / dis

        {{x, stress, i + 1}, dissimilarities, max_iter, similarities_flat_indices,
         similarities_flat, old_stress, stop_value}
      end
  end

  defn remove_main_diag_indices(tensor) do
    n = Nx.axis_size(tensor, 0)

    temp =
      Nx.broadcast(Nx.s64(0), {n})
      |> Nx.indexed_put(Nx.new_axis(0, -1), Nx.s64(1))
      |> Nx.tile([n - 1])

    Nx.iota({n * (n - 1)}) + Nx.cumulative_sum(temp)
    # indices = Nx.iota({n * (n - 1)}) + Nx.cumulative_sum(temp)
    # Nx.take(Nx.flatten(tensor), indices) |> Nx.reshape({n, n - 1})
  end
end
