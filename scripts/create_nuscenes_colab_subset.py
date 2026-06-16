from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


METADATA_FILES = [
    "attribute.json",
    "calibrated_sensor.json",
    "category.json",
    "ego_pose.json",
    "instance.json",
    "log.json",
    "map.json",
    "sample.json",
    "sample_annotation.json",
    "sample_data.json",
    "scene.json",
    "sensor.json",
    "visibility.json",
]


def load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def copy_metadata(metadata_root: Path, output_metadata_root: Path) -> None:
    output_metadata_root.mkdir(parents=True, exist_ok=True)
    for filename in METADATA_FILES:
        source = metadata_root / filename
        if source.exists():
            shutil.copy2(source, output_metadata_root / filename)


def copy_camera_images(
    metadata_root: Path,
    data_root: Path,
    output_root: Path,
    channel: str,
    limit: int,
) -> int:
    sensors = load_json(metadata_root / "sensor.json")
    calibrated_sensors = load_json(metadata_root / "calibrated_sensor.json")
    sample_data_rows = load_json(metadata_root / "sample_data.json")

    sensor_by_token = {row["token"]: row for row in sensors}
    calibrated_to_sensor = {
        row["token"]: sensor_by_token.get(row["sensor_token"], {})
        for row in calibrated_sensors
    }

    copied = 0
    for row in sample_data_rows:
        sensor = calibrated_to_sensor.get(row.get("calibrated_sensor_token", ""), {})
        if sensor.get("channel") != channel:
            continue
        if not row.get("is_key_frame"):
            continue

        relative_path = Path(row["filename"])
        source = data_root / relative_path
        if not source.exists():
            continue

        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

        if copied >= limit:
            break

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small nuScenes subset for Colab batch perception evaluation."
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        required=True,
        help="Path to the nuScenes v1.0-trainval metadata folder.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the nuScenes parent folder containing samples/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output folder to create, for example ~/Desktop/nuscenes_subset.",
    )
    parser.add_argument("--channel", default="CAM_FRONT")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    output_metadata_root = args.output_root / "v1.0-trainval"
    copy_metadata(args.metadata_root, output_metadata_root)
    copied = copy_camera_images(
        metadata_root=args.metadata_root,
        data_root=args.data_root,
        output_root=args.output_root,
        channel=args.channel,
        limit=args.limit,
    )

    print(f"Wrote metadata to: {output_metadata_root}")
    print(f"Copied {copied} {args.channel} images to: {args.output_root / 'samples' / args.channel}")
    print("Upload this output folder or zip it before uploading to Colab/Drive.")


if __name__ == "__main__":
    main()

