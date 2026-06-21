from pathlib import Path

from src.perception_safety_copilot.bdd100k_subset import (
    BddSubsetConfig,
    DEFAULT_CLASS_MAPPING,
    _matches_filters,
    _yolo_row,
)


def _config() -> BddSubsetConfig:
    base = Path("/tmp/bdd100k-test")
    return BddSubsetConfig(
        image_root=base / "images",
        train_json=base / "train.json",
        val_json=base / "val.json",
        output_root=base / "output",
        include_weather={"rainy", "foggy"},
        include_timeofday={"night", "dawn/dusk"},
        include_scene={"city street", "highway"},
        class_mapping=DEFAULT_CLASS_MAPPING,
        max_train=10,
        max_val=5,
        seed=42,
    )


def test_matches_filters_accepts_disturbance_case():
    entry = {
        "attributes": {
            "weather": "rainy",
            "timeofday": "night",
            "scene": "city street",
        }
    }
    assert _matches_filters(entry, _config()) is True


def test_matches_filters_rejects_non_matching_case():
    entry = {
        "attributes": {
            "weather": "clear",
            "timeofday": "daytime",
            "scene": "parking lot",
        }
    }
    assert _matches_filters(entry, _config()) is False


def test_yolo_row_normalizes_box():
    row = _yolo_row({"x1": 10, "y1": 20, "x2": 30, "y2": 60}, class_id=1, width=100, height=200)
    assert row == "1 0.200000 0.200000 0.200000 0.200000"
