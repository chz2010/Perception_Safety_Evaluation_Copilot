from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .evaluation import GroundTruthBox


DEFAULT_CHANNEL = "CAM_FRONT"


@dataclass(frozen=True)
class NuScenesImageSample:
    label: str
    image_path: Path
    sample_token: str
    sample_data_token: str
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
                sample_data_token=sample_data["token"],
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


def quaternion_to_rotation_matrix(rotation: list[float]) -> np.ndarray:
    w, x, y, z = rotation
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def box_3d_corners(
    center: list[float],
    size: list[float],
    rotation: list[float],
) -> np.ndarray:
    width, length, height = size
    x_corners = [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2]
    y_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]
    z_corners = [height / 2, height / 2, height / 2, height / 2, -height / 2, -height / 2, -height / 2, -height / 2]
    corners = np.array([x_corners, y_corners, z_corners], dtype=float)
    return quaternion_to_rotation_matrix(rotation) @ corners + np.array(center, dtype=float).reshape(3, 1)


def transform_points_to_camera(
    points_global: np.ndarray,
    ego_pose: dict[str, Any],
    calibrated_sensor: dict[str, Any],
) -> np.ndarray:
    points_ego = quaternion_to_rotation_matrix(ego_pose["rotation"]).T @ (
        points_global - np.array(ego_pose["translation"], dtype=float).reshape(3, 1)
    )
    points_camera = quaternion_to_rotation_matrix(calibrated_sensor["rotation"]).T @ (
        points_ego - np.array(calibrated_sensor["translation"], dtype=float).reshape(3, 1)
    )
    return points_camera


def project_camera_box_to_image(
    points_camera: np.ndarray,
    camera_intrinsic: list[list[float]],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    visible = points_camera[2, :] > 0.1
    if not visible.any():
        return None

    points = points_camera[:, visible]
    intrinsic = np.array(camera_intrinsic, dtype=float)
    projected = intrinsic @ points
    projected = projected[:2, :] / projected[2:3, :]

    x1 = float(np.clip(projected[0, :].min(), 0, image_width))
    y1 = float(np.clip(projected[1, :].min(), 0, image_height))
    x2 = float(np.clip(projected[0, :].max(), 0, image_width))
    y2 = float(np.clip(projected[1, :].max(), 0, image_height))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def get_ground_truth_boxes_for_camera_sample(
    nuscenes_root: Path,
    sample_token: str,
    sample_data_token: str,
    image_width: int,
    image_height: int,
) -> list[GroundTruthBox]:
    categories = load_json(nuscenes_root / "category.json")
    instances = load_json(nuscenes_root / "instance.json")
    sample_data = load_json(nuscenes_root / "sample_data.json")
    calibrated_sensors = load_json(nuscenes_root / "calibrated_sensor.json")
    ego_pose_path = nuscenes_root / "ego_pose.json"
    if not ego_pose_path.exists():
        return []
    ego_poses = load_json(ego_pose_path)

    sample_data_row = next((row for row in sample_data if row["token"] == sample_data_token), None)
    if sample_data_row is None:
        return []

    calibrated_sensor = {
        row["token"]: row for row in calibrated_sensors
    }.get(sample_data_row["calibrated_sensor_token"])
    ego_pose = {row["token"]: row for row in ego_poses}.get(sample_data_row["ego_pose_token"])
    if calibrated_sensor is None or ego_pose is None:
        return []

    category_by_token = {row["token"]: row["name"] for row in categories}
    instance_category = {
        row["token"]: category_by_token.get(row["category_token"], "unknown")
        for row in instances
    }

    boxes: list[GroundTruthBox] = []
    for annotation in iter_json_array(nuscenes_root / "sample_annotation.json"):
        if annotation.get("sample_token") != sample_token:
            continue

        category_name = instance_category.get(annotation.get("instance_token", ""), "unknown")
        yolo_label = map_nuscenes_category_to_yolo(category_name)
        if yolo_label is None:
            continue

        corners_global = box_3d_corners(annotation["translation"], annotation["size"], annotation["rotation"])
        corners_camera = transform_points_to_camera(corners_global, ego_pose, calibrated_sensor)
        bbox = project_camera_box_to_image(
            corners_camera,
            calibrated_sensor["camera_intrinsic"],
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is not None:
            boxes.append(GroundTruthBox(label=yolo_label, bbox_xyxy=bbox))

    return boxes


def counts_to_ground_truth_text(counts: dict[str, int]) -> str:
    return "\n".join(f"{label}: {count}" for label, count in sorted(counts.items()))
