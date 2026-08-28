defmodule Scholar.Cluster.OPTICSTest do
  use Scholar.Case, async: true

  alias Scholar.Cluster.OPTICS
  doctest OPTICS

  # Reference values from scikit-learn 1.6.1's OPTICS on the same data used
  # in the moduledoc, min_samples=2, max_eps left at its default of inf. With
  # this data that reaches the same reachability graph as eps: 4.5 here, since
  # nothing in it needs capping below that.
  defp x, do: Nx.tensor([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]], type: :f64)

  describe "fit/2 exposes the reachability graph" do
    test "matches scikit-learn's ordering, core distances, reachability and predecessor" do
      model = OPTICS.fit(x(), eps: 4.5, min_samples: 2)

      assert Nx.to_flat_list(model.ordering) == [0, 1, 2, 5, 3, 4]
      assert Nx.to_flat_list(model.predecessor) == [-1, 0, 1, 5, 3, 2]

      core = Nx.to_flat_list(model.core_distances)

      Enum.zip(core, [3.1623, 1.4142, 1.4142, 1.0, 1.0, 4.1231])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-4 end)

      reach = Nx.to_flat_list(model.reachability)
      [first | rest] = reach
      assert first == Nx.Constants.infinity() |> Nx.to_number()

      Enum.zip(rest, [3.1623, 1.4142, 4.1231, 1.0, 5.0])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-4 end)
    end

    # The first point in ordering has nothing to be reached from, which is
    # true of any OPTICS run regardless of the data.
    test "the first point in ordering always has infinite reachability" do
      model = OPTICS.fit(x(), eps: 4.5, min_samples: 2)
      first = Nx.to_number(model.ordering[0])

      assert Nx.to_number(model.reachability[first]) == Nx.to_number(Nx.Constants.infinity())
    end

    test "ordering is a permutation of every sample index" do
      model = OPTICS.fit(x(), eps: 4.5, min_samples: 2)

      assert Nx.to_flat_list(model.ordering) |> Enum.sort() == [0, 1, 2, 3, 4, 5]
    end

    test "every field comes back with one entry per sample" do
      model = OPTICS.fit(x(), eps: 4.5, min_samples: 2)

      for field <- [:labels, :reachability, :core_distances, :ordering, :predecessor] do
        assert Nx.shape(Map.fetch!(model, field)) == {6}
      end
    end
  end
end
