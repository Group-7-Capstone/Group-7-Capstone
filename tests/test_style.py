import matplotlib.pyplot as plt

from viz.style import CATEGORICAL, COMPARISON, MODEL, use_style


def test_use_style_sets_font_and_palette():
    use_style()
    assert plt.rcParams["font.sans-serif"][0] == "Arial"
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"][:2] == [MODEL, COMPARISON]
    assert plt.rcParams["savefig.dpi"] == 300


def test_use_style_registers_colormaps_and_can_run_twice():
    use_style()
    use_style()
    assert {"team_blue", "team_div"} <= set(plt.colormaps())


def test_palette_has_no_duplicates():
    assert len(set(CATEGORICAL)) == len(CATEGORICAL)
