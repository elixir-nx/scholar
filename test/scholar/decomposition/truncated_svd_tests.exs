defmodule Scholar.Decomposition.TruncatedSVDTest do
  use Scholar.Case, async: true
  alias Scholar.Decomposition.TruncatedSVD
  doctest TruncatedSVD

  defp key do
    Nx.Random.key(1)
  end

  defp x do
    key = key()
    {x, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0.0, 0.0, 0.0, 0.0]), Nx.tensor([[3.0, 2.0, 1.0, 9.0], [1.0, 2.0, 3.0, 8.2], [1.3, 1.0, 2.2, 2.4], [1.8, 1.0, 2.0, 2.9]]), shape: {50}, type: :f32)
    x
  end
  # key = Nx.Random.key(1)
  # {x, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0.0, 0.0, 0.0, 0.0]), Nx.tensor([[3.0, 2.0, 1.0, 9.0], [1.0, 2.0, 3.0, 8.2], [1.3, 1.0, 2.2, 2.4], [1.8, 1.0, 2.0, 2.9]]), shape: {50}, type: :f32)
  # tsvd = Scholar.Decomposition.TruncatedSVD.fit_transform(x, num_components: 2, key: key)


  test "fit test - all default options" do
    key = key()
    x = x()
  end

  test "fit_transform test - all default options" do
    #tsvd = Scholar.Decomposition.TruncatedSVD.fit_transform(x, num_components: 2, key: key)
  end

  test "fit_transform test - :num_components" do
    
  end

  test "fit_transform test - :num_oversamples" do
    
  end

  test "fit_transform test - :num_iters" do
    
  end
end
