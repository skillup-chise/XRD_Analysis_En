"""Lattice inputs must show the stored cell when a crystal system reveals them."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _select(at: AppTest, label: str):
    for box in at.selectbox:
        if box.label == label:
            return box
    raise AssertionError(f"No selectbox labeled {label!r}")


def _number(at: AppTest, label: str):
    matches = [box for box in at.number_input if box.label == label]
    assert len(matches) == 1, [box.label for box in at.number_input]
    return matches[0]


def test_hidden_lattice_input_returns_with_the_stored_cell():
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()

    _select(at, "Catalog").set_value("Custom (Other)").run()
    assert _number(at, "a (Å)").proto.default == 4.0
    assert [box.label for box in at.number_input if box.label == "c (Å)"] == []

    _select(at, "Crystal system").set_value("Tetragonal").run()
    assert _number(at, "a (Å)").proto.default == 4.0
    assert _number(at, "c (Å)").proto.default == 4.0

    _select(at, "Crystal system").set_value("Monoclinic").run()
    assert _number(at, "a (Å)").proto.default == 4.0
    assert _number(at, "b (Å)").proto.default == 4.0
    assert _number(at, "c (Å)").proto.default == 4.0
    assert _number(at, "beta (°)").proto.default == 90.0

    _select(at, "Crystal system").set_value("Cubic").run()
    assert [box.label for box in at.number_input if box.label == "c (Å)"] == []
    _select(at, "Crystal system").set_value("Hexagonal").run()
    assert _number(at, "a (Å)").proto.default == 4.0
    assert _number(at, "c (Å)").proto.default == 4.0
