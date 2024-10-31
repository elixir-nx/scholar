defmodule Scholar.CrossDecomposition.PLSSVDTest do
  use Scholar.Case, async: true
  alias Scholar.CrossDecomposition.PLSSVD
  doctest PLSSVD

  defp x do
    Nx.tensor([
      [0.0, 0.0, 1.0, 16.0],
      [1.0, 0.0, 0.0, 25.2],
      [2.0, 2.0, 2.0, -2.3],
      [2.0, 5.0, 4.0, 4.5],
      [5.0, -2.0, 3.3, 4.5]
    ])
  end

  defp y do
    Nx.tensor([
      [0.1, -0.2, 3.0],
      [0.9, 1.1, 5.1],
      [6.2, 5.9, 2.5],
      [11.9, 12.3, -6.0],
      [7.6, 1.8, 4.9]
    ])
  end

  defp y_1d do
    Nx.tensor([0.1, -0.2, 3.0, 6.9, 3])
  end

  test "fit test" do
    model = Scholar.CrossDecomposition.PLSSVD.fit(x(), y())

    assert_all_close(
      model.x_mean,
      Nx.tensor([2.0, 1.0, 2.059999942779541, 9.579999923706055]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.y_mean,
      Nx.tensor([5.339999675750732, 4.179999828338623, 1.899999976158142]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.x_std,
      Nx.tensor([1.8708287477493286, 2.6457512378692627, 1.6334013938903809, 10.931011199951172]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.y_std,
      Nx.tensor([4.90030574798584, 5.08005952835083, 4.561249732971191]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.x_weights,
      Nx.tensor([
        [0.17879533767700195, 0.7447080016136169],
        [0.6228733062744141, -0.5843358635902405],
        [0.6137028336524963, 0.1790202558040619],
        [-0.4510321617126465, -0.26816627383232117]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(
      model.y_weights,
      Nx.tensor([
        [0.6292941570281982, 0.5848351716995239, -0.5118170976638794],
        [0.7398861646652222, -0.2493150532245636, 0.6248283386230469]
      ]),
      atol: 1.0e-3
    )
  end

  test "transform test" do
    model = Scholar.CrossDecomposition.PLSSVD.fit(x(), y())
    {x_transformed, y_transformed} = Scholar.CrossDecomposition.PLSSVD.transform(model, x(), y())

    assert_all_close(
      x_transformed,
      Nx.tensor([
        [-1.0897283554077148, -0.8489431142807007],
        [-1.7494868040084839, -0.7861797213554382],
        [0.703069806098938, 0.06401326507329941],
        [1.8802037239074707, -0.5461838245391846],
        [0.25594159960746765, 2.117293357849121]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(
      y_transformed,
      Nx.tensor([
        [-1.3005900382995605, -0.42553290724754333],
        [-1.2838343381881714, -0.08087197691202164],
        [0.24112752079963684, 0.12762844562530518],
        [2.6636931896209717, -0.49021831154823303],
        [-0.3203960657119751, 0.8689947128295898]
      ]),
      atol: 1.0e-3
    )
  end

  test "fit_transform test - all options are default" do
    {x_transformed, y_transformed} = Scholar.CrossDecomposition.PLSSVD.fit_transform(x(), y())

    assert_all_close(
      x_transformed,
      Nx.tensor([
        [-1.0897283554077148, -0.8489431142807007],
        [-1.7494868040084839, -0.7861797213554382],
        [0.703069806098938, 0.06401326507329941],
        [1.8802037239074707, -0.5461838245391846],
        [0.25594159960746765, 2.117293357849121]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(
      y_transformed,
      Nx.tensor([
        [-1.3005900382995605, -0.42553290724754333],
        [-1.2838343381881714, -0.08087197691202164],
        [0.24112752079963684, 0.12762844562530518],
        [2.6636931896209717, -0.49021831154823303],
        [-0.3203960657119751, 0.8689947128295898]
      ]),
      atol: 1.0e-3
    )
  end

  test "fit_transform test - :num_components set to 1" do
    {x_transformed, y_transformed} =
      Scholar.CrossDecomposition.PLSSVD.fit_transform(x(), y(), num_components: 1)

    assert_all_close(
      x_transformed,
      Nx.tensor([
        [-1.0897283554077148],
        [-1.7494868040084839],
        [0.703069806098938],
        [1.8802037239074707],
        [0.25594159960746765]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(
      y_transformed,
      Nx.tensor([
        [-1.3005900382995605],
        [-1.2838343381881714],
        [0.24112752079963684],
        [2.6636931896209717],
        [-0.3203960657119751]
      ]),
      atol: 1.0e-3
    )
  end

  test "fit_transform test - y is has only one dimension" do
    {x_transformed, y_transformed} =
      Scholar.CrossDecomposition.PLSSVD.fit_transform(x(), y_1d(), num_components: 1)

    assert_all_close(
      x_transformed,
      Nx.tensor([
        [-1.2138643264770508],
        [-1.868216872215271],
        [0.703800618648529],
        [1.7553009986877441],
        [0.6229796409606934]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(
      y_transformed,
      Nx.tensor([
        [-0.8578669428825378],
        [-0.9624848365783691],
        [0.15343964099884033],
        [1.5134726762771606],
        [0.15343964099884033]
      ]),
      atol: 1.0e-3
    )
  end

  test "fit_transform test - :scale is set to :false" do
    {x_transformed, y_transformed} =
      Scholar.CrossDecomposition.PLSSVD.fit_transform(x(), y(), scale: false)

    assert_all_close(
      x_transformed,
      Nx.tensor([
        [6.641565322875977, 1.5491820573806763],
        [15.36169719696045, 3.2503585815429688],
        [-11.394588470458984, -2.017521619796753],
        [-6.2775702476501465, 2.303945779800415],
        [-4.3311028480529785, -5.085964679718018]
      ]),
      atol: 1.0e-3
    )

    assert_all_close(
      y_transformed,
      Nx.tensor([
        [6.744043827056885, 1.1535897254943848],
        [6.1893134117126465, -0.3978065252304077],
        [-1.4090275764465332, -0.40731552243232727],
        [-12.453459739685059, 3.961534023284912],
        [0.9291285872459412, -4.310001850128174]
      ]),
      atol: 1.0e-3
    )
  end
end
