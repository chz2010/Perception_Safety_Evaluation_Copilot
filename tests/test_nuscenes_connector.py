from src.perception_safety_copilot.nuscenes_connector import (
    counts_to_ground_truth_text,
    map_nuscenes_category_to_yolo,
)


def test_map_nuscenes_categories_to_yolo_labels():
    assert map_nuscenes_category_to_yolo("human.pedestrian.adult") == "person"
    assert map_nuscenes_category_to_yolo("vehicle.car") == "car"
    assert map_nuscenes_category_to_yolo("vehicle.bus.rigid") == "bus"
    assert map_nuscenes_category_to_yolo("vehicle.truck") == "truck"
    assert map_nuscenes_category_to_yolo("animal") is None


def test_counts_to_ground_truth_text_is_stable():
    assert counts_to_ground_truth_text({"car": 2, "person": 1}) == "car: 2\nperson: 1"

