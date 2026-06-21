from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO model on the local BDD100K YOLO-format dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("training/bdd100k_yolo_local.yaml"),
        help="Path to the YOLO dataset YAML.",
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Base Ultralytics model checkpoint to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Lower this if memory is tight.")
    parser.add_argument("--device", default="cpu", help="Training device, for example cpu, mps, or 0.")
    parser.add_argument(
        "--project",
        default="runs/bdd100k_training",
        help="Folder for Ultralytics training outputs.",
    )
    parser.add_argument(
        "--name",
        default="yolo11s_disturbance_ft",
        help="Run name inside the training project folder.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    return parser


def _prepare_dataset_yaml(data_path: Path) -> Path:
    lines = data_path.read_text(encoding="utf-8").splitlines()
    resolved_lines: list[str] = []

    for line in lines:
        if line.startswith("path:"):
            raw_value = line.split(":", 1)[1].strip().strip("\"'")
            dataset_root = Path(raw_value)
            if not dataset_root.is_absolute():
                dataset_root = (data_path.parent / dataset_root).resolve()
            resolved_lines.append(f"path: {dataset_root}")
        else:
            resolved_lines.append(line)

    temp_dir = Path(tempfile.mkdtemp(prefix="bdd100k_yolo_"))
    temp_yaml = temp_dir / data_path.name
    temp_yaml.write_text("\n".join(resolved_lines) + "\n", encoding="utf-8")
    return temp_yaml


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Activate the project virtualenv and run "
            "`pip install -r requirements.txt` first."
        ) from exc

    data_path = args.data.resolve()
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")
    prepared_data_path = _prepare_dataset_yaml(data_path)
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (project_root / project_path).resolve()

    model = YOLO(args.model)
    model.train(
        data=str(prepared_data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_path),
        name=args.name,
        workers=args.workers,
        exist_ok=True,
    )

    print("Training complete.")
    print(f"Run outputs: {project_path / args.name}")
    print(f"Dataset config used: {prepared_data_path}")


if __name__ == "__main__":
    main()
