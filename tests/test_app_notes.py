"""Experiment notes are recorded with a loaded scan and stay out of the calculation."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _number(at: AppTest, label: str):
    matches = [box for box in at.number_input if box.label == label]
    assert len(matches) == 1, [box.label for box in at.number_input]
    return matches[0]


def _metric_map(at: AppTest) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in at.metric:
        label = getattr(item, "label", None) or item.proto.label
        labels[label] = item.value
    return labels


def test_experiment_notes_are_shown_with_the_loaded_material():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Ti2AlN"

    _number(at, "Stirrer size (mm)").set_value(42.0)
    _number(at, "Rotation speed (rpm)").set_value(300.0)
    _number(at, "Flask size (mL)").set_value(250.0)
    note = next(box for box in at.text_area if box.label == "Short note")
    note.set_value("Argon, 2 h")
    at.run()
    assert not at.exception

    button = next(box for box in at.button if box.label == "Load Ti2AlN example")
    button.click().run()
    assert not at.exception

    assert at.title[0].value == "Ti2AlN"
    metrics = _metric_map(at)
    assert metrics["Stirrer size"] == "42 mm"
    assert metrics["Rotation speed"] == "300 rpm"
    assert metrics["Flask size"] == "250 mL"
    captions = "\n".join(item.value for item in at.caption)
    assert "Recorded for Ti2AlN" in captions
    assert "experiment notes only" in captions
    assert "do not change peak positions" in captions
    assert any("Argon, 2 h" in item.value for item in at.caption)
