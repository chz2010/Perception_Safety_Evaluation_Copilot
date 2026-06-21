from __future__ import annotations

import argparse
from pathlib import Path

from src.perception_safety_copilot.bdd100k_subset import (
    BddSubsetConfig,
    DEFAULT_CLASS_MAPPING,
    create_bdd100k_yolo_subset,
)


def _csv_to_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a disturbance-focused YOLO subset from BDD100K.")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("archive (1)"),
        help="Folder containing bdd100k/, bdd100k_labels_release/, and optionally bdd100k_seg/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training/bdd100k_disturbance_subset"),
        help="Output folder for YOLO-formatted images, labels, and reports.",
    )
    parser.add_argument(
        "--weather",
        default="rainy,foggy,snowy,overcast",
        help="Comma-separated weather values to include.",
    )
    parser.add_argument(
        "--timeofday",
        default="night,dawn/dusk",
        help="Comma-separated time-of-day values to include.",
    )
    parser.add_argument(
        "--scene",
        default="city street,highway,residential,tunnel",
        help="Optional comma-separated scene values to include. Use empty string for all scenes.",
    )
    parser.add_argument("--max-train", type=int, default=4000, help="Maximum number of train images to keep.")
    parser.add_argument("--max-val", type=int, default=1000, help="Maximum number of val images to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset selection.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    archive_root = args.archive_root
    image_root = archive_root / "bdd100k" / "bdd100k" / "images" / "100k"
    labels_root = archive_root / "bdd100k_labels_release" / "bdd100k" / "labels"

    config = BddSubsetConfig(
        image_root=image_root,
        train_json=labels_root / "bdd100k_labels_images_train.json",
        val_json=labels_root / "bdd100k_labels_images_val.json",
        output_root=args.output_root,
        include_weather=_csv_to_set(args.weather),
        include_timeofday=_csv_to_set(args.timeofday),
        include_scene=_csv_to_set(args.scene) if args.scene.strip() else None,
        class_mapping=DEFAULT_CLASS_MAPPING,
        max_train=args.max_train,
        max_val=args.max_val,
        seed=args.seed,
    )

    summaries = create_bdd100k_yolo_subset(config)
    print(f"Created subset at: {config.output_root}")
    for summary in summaries:
        print(
            f"{summary['split']}: selected={summary['selected_images']} "
            f"missing={summary['skipped_missing_images']} "
            f"no_relevant_labels={summary['skipped_without_relevant_labels']}"
        )
    print(f"Summary report: {config.output_root / 'subset_summary.md'}")
    print(f"YOLO config: {config.output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
