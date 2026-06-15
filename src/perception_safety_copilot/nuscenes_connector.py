from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


DEFAULT_CHANNEL = "CAM_FRONT"


@dataclass(frozen=True)
class NuScenesImageSample:
    label: str
    image_path: Path
    sample_token: str
    scene_name: str
    scene_description: str
    channel: str


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False

    with path.open(encoding="utf-8") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk and not buffer.strip():
                break
            buffer += chunk

            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"Expected JSON array in {path}")
                    buffer = buffer[1:]
                    started = True
                    continue

                buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()

                try:
                    obj, index = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break

                yield obj
                buffer = buffer[index:]

            if not chunk:
                break


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def is_nuscenes_root(root: Path) -> bool:
    required = [
        "sample.json",
        "sample_data.json",
        "sample_annotation.json",
        "sensor.json",
        "calibrated_sensor.json",
        "scene.json",
        "instance.json",
        "category.json",
    ]
    return root.exists() and all((root / name).exists() for name in required)


def map_nuscenes_category_to_yolo(category_name: str) -> str | None:
    if category_name.startswith("human.pedestrian"):
        return "person"
    if category_name == "vehicle.car":
        return "car"
    if category_name.startswith("vehicle.truck"):
        return "truck"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name == "vehicle.motorcycle":
        return "motorcycle"
    if category_name == "vehicle.bicycle":
        return "bicycle"
    if category_name == "movable_object.trafficcone":
        return "traffic cone"
    return None


def discover_camera_samples(
    metadata_root: Path,
    channel: str = DEFAULT_CHANNEL,
    limit: int = 25,
    data_root: Path | None = None,
) -> list[NuScenesImageSample]:
    if not is_nuscenes_root(metadata_root):
        raise FileNotFoundError(f"{metadata_root} does not look like a nuScenes metadata root")

    sensors = load_json(metadata_root / "sensor.json")
    calibrated_sensors = load_json(metadata_root / "calibrated_sensor.json")
    scenes = load_json(metadata_root / "scene.json")
    samples = load_json(metadata_root / "sample.json")

    sensor_by_token = {row["token"]: row for row in sensors}
    calibrated_to_sensor = {
        row["token"]: sensor_by_token.get(row["sensor_token"], {})
        for row in calibrated_sensors
    }
    sample_to_scene = {row["token"]: row.get("scene_token") for row in samples}
    scene_by_token = {row["token"]: row for row in scenes}
    image_roots = [root for root in [data_root, metadata_root, metadata_root.parent] if root is not None]

    discovered: list[NuScenesImageSample] = []
    for sample_data in iter_json_array(metadata_root / "sample_data.json"):
        sensor = calibrated_to_sensor.get(sample_data.get("calibrated_sensor_token", ""), {})
        if sensor.get("channel") != channel:
            continue
        if not sample_data.get("is_key_frame"):
            continue

        filename = sample_data.get("filename", "")
        image_path = next((root / filename for root in image_roots if (root / filename).exists()), None)
        if image_path is None:
            continue

        sample_token = sample_data["sample_token"]
        scene = scene_by_token.get(sample_to_scene.get(sample_token, ""), {})
        scene_name = scene.get("name", "unknown_scene")
        scene_description = scene.get("description", "")
        discovered.append(
            NuScenesImageSample(
                label=f"{scene_name} | {image_path.name}",
                image_path=image_path,
                sample_token=sample_token,
                scene_name=scene_name,
                scene_description=scene_description,
                channel=channel,
            )
        )
        if len(discovered) >= limit:
            break

    return discovered


def get_expected_counts_for_sample(nuscenes_root: Path, sample_token: str) -> dict[str, int]:
    categories = load_json(nuscenes_root / "category.json")
    instances = load_json(nuscenes_root / "instance.json")

    category_by_token = {row["token"]: row["name"] for row in categories}
    instance_category = {
        row["token"]: category_by_token.get(row["category_token"], "unknown")
        for row in instances
    }

    expected: dict[str, int] = {}
    for annotation in iter_json_array(nuscenes_root / "sample_annotation.json"):
        if annotation.get("sample_token") != sample_token:
            continue
        category_name = instance_category.get(annotation.get("instance_token", ""), "unknown")
        yolo_label = map_nuscenes_category_to_yolo(category_name)
        if yolo_label is None:
            continue
        expected[yolo_label] = expected.get(yolo_label, 0) + 1
    return expected


def counts_to_ground_truth_text(counts: dict[str, int]) -> str:
    return "\n".join(f"{label}: {count}" for label, count in sorted(counts.items()))
