from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


DEFAULT_CLASS_MAPPING = {
    "person": 0,
    "car": 1,
    "truck": 2,
    "bus": 3,
    "bike": 4,
    "motor": 5,
    "traffic light": 6,
}


@dataclass(frozen=True)
class BddSubsetConfig:
    image_root: Path
    train_json: Path
    val_json: Path
    output_root: Path
    include_weather: set[str]
    include_timeofday: set[str]
    include_scene: set[str] | None
    class_mapping: dict[str, int]
    max_train: int | None
    max_val: int | None
    seed: int


def normalize_text(value: str | None) -> str:
    return (value or "undefined").strip().lower()


def _matches_filters(entry: dict, config: BddSubsetConfig) -> bool:
    attributes = entry.get("attributes", {})
    weather = normalize_text(attributes.get("weather"))
    timeofday = normalize_text(attributes.get("timeofday"))
    scene = normalize_text(attributes.get("scene"))

    if config.include_weather and weather not in config.include_weather:
        return False
    if config.include_timeofday and timeofday not in config.include_timeofday:
        return False
    if config.include_scene and scene not in config.include_scene:
        return False
    return True


def _yolo_row(box2d: dict, class_id: int, width: int, height: int) -> str | None:
    x1 = float(box2d["x1"])
    y1 = float(box2d["y1"])
    x2 = float(box2d["x2"])
    y2 = float(box2d["y2"])
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0:
        return None

    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def _extract_label_rows(entry: dict, class_mapping: dict[str, int], image_width: int, image_height: int) -> tuple[list[str], Counter[str]]:
    rows: list[str] = []
    counts: Counter[str] = Counter()
    for label in entry.get("labels", []):
        category = normalize_text(label.get("category"))
        if category not in class_mapping:
            continue
        box2d = label.get("box2d")
        if not box2d:
            continue
        row = _yolo_row(box2d, class_mapping[category], image_width, image_height)
        if row is None:
            continue
        rows.append(row)
        counts[category] += 1
    return rows, counts


def _prepare_split(
    split_name: str,
    json_path: Path,
    image_dir: Path,
    labels_dir: Path,
    config: BddSubsetConfig,
    limit: int | None,
) -> dict:
    entries = json.loads(json_path.read_text())
    filtered = [entry for entry in entries if _matches_filters(entry, config)]

    rng = random.Random(config.seed + (0 if split_name == "train" else 1))
    rng.shuffle(filtered)
    if limit is not None:
        filtered = filtered[:limit]

    image_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    selected_images = 0
    skipped_missing_images = 0
    skipped_without_relevant_labels = 0
    weather_counts: Counter[str] = Counter()
    timeofday_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    source_split_dir = config.image_root / split_name

    for entry in filtered:
        image_name = entry["name"]
        source_image = source_split_dir / image_name
        if not source_image.exists():
            skipped_missing_images += 1
            continue

        with Image.open(source_image) as image:
            width, height = image.size

        rows, row_counts = _extract_label_rows(entry, config.class_mapping, width, height)
        if not rows:
            skipped_without_relevant_labels += 1
            continue

        shutil.copy2(source_image, image_dir / image_name)
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        attributes = entry.get("attributes", {})
        weather_counts[normalize_text(attributes.get("weather"))] += 1
        timeofday_counts[normalize_text(attributes.get("timeofday"))] += 1
        scene_counts[normalize_text(attributes.get("scene"))] += 1
        class_counts.update(row_counts)
        selected_images += 1

    return {
        "split": split_name,
        "requested_entries": len(filtered),
        "selected_images": selected_images,
        "skipped_missing_images": skipped_missing_images,
        "skipped_without_relevant_labels": skipped_without_relevant_labels,
        "weather_counts": dict(weather_counts),
        "timeofday_counts": dict(timeofday_counts),
        "scene_counts": dict(scene_counts),
        "class_counts": dict(class_counts),
    }


def _write_data_yaml(output_root: Path, class_mapping: dict[str, int]) -> None:
    names = {class_id: name for name, class_id in class_mapping.items()}
    lines = [
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for class_id in sorted(names):
        lines.append(f"  {class_id}: {names[class_id]}")
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_report(output_root: Path, config: BddSubsetConfig, summaries: list[dict]) -> None:
    summary = {
        "filters": {
            "weather": sorted(config.include_weather),
            "timeofday": sorted(config.include_timeofday),
            "scene": sorted(config.include_scene) if config.include_scene else [],
        },
        "class_mapping": config.class_mapping,
        "splits": summaries,
    }
    (output_root / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# BDD100K Disturbance Subset Summary",
        "",
        "## Filters",
        f"- Weather: {', '.join(sorted(config.include_weather)) or 'All'}",
        f"- Time of day: {', '.join(sorted(config.include_timeofday)) or 'All'}",
        f"- Scene: {', '.join(sorted(config.include_scene)) if config.include_scene else 'All'}",
        "",
        "## Class Mapping",
    ]
    for label, class_id in sorted(config.class_mapping.items(), key=lambda item: item[1]):
        lines.append(f"- {class_id}: {label}")

    for split_summary in summaries:
        lines.extend(
            [
                "",
                f"## {split_summary['split'].title()} Split",
                f"- Selected images: {split_summary['selected_images']}",
                f"- Missing images skipped: {split_summary['skipped_missing_images']}",
                f"- Images without relevant labels skipped: {split_summary['skipped_without_relevant_labels']}",
                "",
                "### Weather Counts",
            ]
        )
        for label, count in sorted(split_summary["weather_counts"].items()):
            lines.append(f"- {label}: {count}")
        lines.extend(["", "### Time-of-Day Counts"])
        for label, count in sorted(split_summary["timeofday_counts"].items()):
            lines.append(f"- {label}: {count}")
        lines.extend(["", "### Scene Counts"])
        for label, count in sorted(split_summary["scene_counts"].items()):
            lines.append(f"- {label}: {count}")
        lines.extend(["", "### Class Counts"])
        for label, count in sorted(split_summary["class_counts"].items()):
            lines.append(f"- {label}: {count}")

    (output_root / "subset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_bdd100k_yolo_subset(config: BddSubsetConfig) -> list[dict]:
    if config.output_root.exists():
        shutil.rmtree(config.output_root)
    (config.output_root / "images").mkdir(parents=True, exist_ok=True)
    (config.output_root / "labels").mkdir(parents=True, exist_ok=True)

    summaries = []
    summaries.append(
        _prepare_split(
            "train",
            config.train_json,
            config.output_root / "images" / "train",
            config.output_root / "labels" / "train",
            config,
            config.max_train,
        )
    )
    summaries.append(
        _prepare_split(
            "val",
            config.val_json,
            config.output_root / "images" / "val",
            config.output_root / "labels" / "val",
            config,
            config.max_val,
        )
    )

    _write_data_yaml(config.output_root, config.class_mapping)
    _write_summary_report(config.output_root, config, summaries)
    return summaries
