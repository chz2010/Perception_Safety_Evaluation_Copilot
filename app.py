from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.perception_safety_copilot.detection import (
    detections_to_records,
    draw_detections,
    load_yolo_model,
    run_yolo_detection,
)
from src.perception_safety_copilot.evaluation import Detection, evaluate_detections
from src.perception_safety_copilot.nuscenes_connector import (
    DEFAULT_CHANNEL,
    counts_to_ground_truth_text,
    discover_camera_samples,
    get_expected_counts_for_sample,
    is_nuscenes_root,
)
from src.perception_safety_copilot.project1_bridge import (
    DEFAULT_NUSCENES_PROFILE,
    DEFAULT_PROJECT1_DIR,
    build_project1_context_section,
    build_project1_standards_section,
    load_nuscenes_safety_profile,
    load_project1_standard_context,
)
from src.perception_safety_copilot.reporting import generate_markdown_report
from src.perception_safety_copilot.safety_lens import (
    evaluate_safety_lens,
    generate_safety_report,
    infer_scenario_tags,
    parse_expected_objects,
)
from src.perception_safety_copilot.storage import (
    DEFAULT_DB_PATH,
    load_recent_evaluations,
    save_evaluation,
)


OUTPUTS_DIR = Path("outputs")
BATCH_DETECTIONS_GLOB = "perception_yolo_results*.csv"
BATCH_SUMMARY_GLOB = "perception_eval_summary*.csv"
BATCH_REPORT_GLOB = "perception_failure_report*.md"
INTERNAL_CANDIDATE_THRESHOLD = 0.25


st.set_page_config(
    page_title="Perception Safety Evaluation Copilot",
    page_icon=":material/directions_car:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading YOLO model...")
def get_model(model_name: str):
    return load_yolo_model(model_name)


@st.cache_data(show_spinner=False)
def get_nuscenes_samples(metadata_root: str, data_root: str, channel: str, limit: int):
    resolved_data_root = Path(data_root) if data_root else None
    return discover_camera_samples(
        Path(metadata_root),
        channel=channel,
        limit=limit,
        data_root=resolved_data_root,
    )


@st.cache_data(show_spinner=False)
def get_nuscenes_expected_counts(nuscenes_root: str, sample_token: str):
    return get_expected_counts_for_sample(Path(nuscenes_root), sample_token)


@st.cache_data(show_spinner=False)
def get_nuscenes_ground_truth_boxes(
    nuscenes_root: str,
    sample_token: str,
    sample_data_token: str,
    image_width: int,
    image_height: int,
):
    try:
        from src.perception_safety_copilot.nuscenes_connector import (
            get_ground_truth_boxes_for_camera_sample,
        )
    except ImportError:
        return []
    return get_ground_truth_boxes_for_camera_sample(
        Path(nuscenes_root),
        sample_token,
        sample_data_token,
        image_width,
        image_height,
    )


def metrics_to_dict(result) -> dict:
    iou = result.iou_evaluation
    return {
        "detected_total": sum(result.detected_counts.values()),
        "missed_total": sum(result.missed_objects.values()),
        "false_positive_total": sum(result.false_positives.values()),
        "low_confidence_total": len(result.low_confidence_detections),
        "precision": result.precision,
        "recall": result.recall,
        "iou_threshold": None if iou is None else iou.threshold,
        "iou_matched_detections": None if iou is None else iou.matched_detections,
        "iou_unmatched_detections": None if iou is None else iou.unmatched_detections,
        "iou_unmatched_ground_truth": None if iou is None else iou.unmatched_ground_truth,
        "iou_precision": None if iou is None else iou.precision,
        "iou_recall": None if iou is None else iou.recall,
        "mean_iou": None if iou is None else iou.mean_iou,
        "map50": result.map50,
        "map50_95": result.map50_95,
        "detected_counts": result.detected_counts,
        "expected_counts": result.expected_counts,
        "missed_objects": result.missed_objects,
        "false_positives": result.false_positives,
    }


def render_detection_table(records: list[dict]) -> None:
    if not records:
        st.info("No objects detected at the selected confidence threshold.")
        return

    table = pd.DataFrame(records)
    table["confidence"] = table["confidence"].map(lambda value: round(value, 3))
    table[["x1", "y1", "x2", "y2"]] = pd.DataFrame(table["bbox_xyxy"].tolist(), index=table.index)
    table = table.drop(columns=["bbox_xyxy"])
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_recent_history() -> None:
    recent = load_recent_evaluations(DEFAULT_DB_PATH, limit=10)
    if not recent:
        st.caption("No saved evaluations yet.")
        return

    rows = []
    for item in recent:
        metrics = item["metrics"]
        rows.append(
            {
                "id": item["id"],
                "created_at": item["created_at"],
                "scenario": item["scenario_name"] or "Unspecified",
                "image": item["image_name"],
                "detected": metrics.get("detected_total", 0),
                "missed": metrics.get("missed_total", 0),
                "low_conf": metrics.get("low_confidence_total", 0),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def parse_count_dict(value) -> dict[str, int]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(label): int(count) for label, count in parsed.items()}


def aggregate_count_column(series: pd.Series) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for value in series:
        for label, count in parse_count_dict(value).items():
            counts[label] = counts.get(label, 0) + count
    if not counts:
        return pd.DataFrame(columns=["label", "count"])
    return (
        pd.DataFrame([{"label": label, "count": count} for label, count in counts.items()])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def aggregate_count_dict(series: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series:
        for label, count in parse_count_dict(value).items():
            counts[label] = counts.get(label, 0) + count
    return counts


def run_id_from_output_path(path: Path, prefix: str, suffix: str) -> str:
    if path.name == f"{prefix}{suffix}":
        return "latest"
    return path.name.removeprefix(f"{prefix}_").removesuffix(suffix)


def load_csv_outputs(pattern: str, prefix: str) -> pd.DataFrame:
    frames = []
    for path in sorted(OUTPUTS_DIR.glob(pattern)):
        frame = pd.read_csv(path)
        frame["run_id"] = run_id_from_output_path(path, prefix, ".csv")
        frame["source_file"] = path.name
        frame["source_mtime"] = path.stat().st_mtime
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def detections_from_dataframe(frame: pd.DataFrame) -> list[Detection]:
    detections: list[Detection] = []
    for row in frame.to_dict("records"):
        detections.append(
            Detection(
                label=str(row["label"]),
                confidence=float(row["confidence"]),
                bbox_xyxy=(
                    float(row.get("x1", 0.0)),
                    float(row.get("y1", 0.0)),
                    float(row.get("x2", 0.0)),
                    float(row.get("y2", 0.0)),
                ),
                image_id=row.get("image"),
                model_name=row.get("model"),
            )
        )
    return detections


def summarize_runs(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    aggregations = {
        "images": ("image", "count"),
        "expected": ("expected_total", "sum"),
        "ground_truth_boxes": ("ground_truth_box_total", "sum"),
        "missed": ("missed_total", "sum"),
        "false_positives": ("false_positive_total", "sum"),
        "low_confidence": ("low_confidence_total", "sum"),
        "mean_precision": ("precision", "mean"),
        "mean_recall": ("recall", "mean"),
    }
    optional_aggregations = {
        "mean_iou": ("mean_iou", "mean"),
        "map50": ("map50", "mean"),
        "map50_95": ("map50_95", "mean"),
        "iou_precision": ("iou_precision", "mean"),
        "iou_recall": ("iou_recall", "mean"),
    }
    aggregations.update(
        {
            output_column: aggregation
            for output_column, aggregation in optional_aggregations.items()
            if aggregation[0] in summary_df
        }
    )

    run_summary = summary_df.groupby("run_id", as_index=False).agg(**aggregations)
    if "source_mtime" in summary_df:
        run_source = (
            summary_df.groupby("run_id", as_index=False)["source_mtime"]
            .max()
            .rename(columns={"source_mtime": "updated_at"})
        )
        run_summary = run_summary.merge(run_source, on="run_id", how="left")

    preferred_sort = "map50" if "map50" in run_summary else "mean_recall"
    return run_summary.sort_values(preferred_sort, ascending=False)


def pick_latest_path(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def render_batch_dashboard() -> None:
    st.subheader("Batch Evaluation Results")
    st.caption("Colab-generated perception evaluation artifacts from `outputs/`.")

    summary_df_all = load_csv_outputs(BATCH_SUMMARY_GLOB, "perception_eval_summary")
    detections_df_all = load_csv_outputs(BATCH_DETECTIONS_GLOB, "perception_yolo_results")

    if summary_df_all.empty or detections_df_all.empty:
        st.info(
            "Batch outputs were not found yet. Run the Colab notebook, then place "
            "`perception_eval_summary*.csv` and `perception_yolo_results*.csv` in `outputs/`."
        )
        return

    run_ids = sorted(set(summary_df_all["run_id"]) | set(detections_df_all["run_id"]))
    selected_runs = st.multiselect("Batch runs", run_ids, default=run_ids)
    if not selected_runs:
        st.warning("Select at least one batch run.")
        return

    summary_df = summary_df_all[summary_df_all["run_id"].isin(selected_runs)].copy()
    detections_df = detections_df_all[detections_df_all["run_id"].isin(selected_runs)].copy()
    run_summary = summarize_runs(summary_df)

    st.markdown("#### Run Summary")
    if run_summary.empty:
        st.info("No batch rows available for the selected runs.")
    else:
        display_columns = [
            "run_id",
            "images",
            "expected",
            "ground_truth_boxes",
            "missed",
            "false_positives",
            "low_confidence",
            "mean_precision",
            "mean_recall",
        ]
        display_columns.extend([column for column in ["mean_iou", "map50", "map50_95", "iou_precision", "iou_recall", "updated_at"] if column in run_summary.columns])
        st.dataframe(run_summary[display_columns], use_container_width=True, hide_index=True)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Images", f"{len(summary_df):,}")
    metric_cols[1].metric("Detections", f"{len(detections_df):,}")
    metric_cols[2].metric("Expected", f"{int(summary_df['expected_total'].sum()):,}")
    metric_cols[3].metric("Missed", f"{int(summary_df['missed_total'].sum()):,}")
    metric_cols[4].metric("Low Conf.", f"{int(summary_df['low_confidence_total'].sum()):,}")
    metric_cols[5].metric("Mean Recall", f"{summary_df['recall'].mean():.2f}")

    map_columns = [column for column in ["map50", "map50_95", "mean_iou"] if column in summary_df]
    if map_columns:
        map_cols = st.columns(len(map_columns))
        for index, column in enumerate(map_columns):
            value = pd.to_numeric(summary_df[column], errors="coerce").mean()
            label = {"map50": "mAP50", "map50_95": "mAP50-95", "mean_iou": "Mean IoU"}[column]
            map_cols[index].metric(label, "N/A" if pd.isna(value) else f"{value:.3f}")

    if len(selected_runs) > 1:
        st.markdown("#### Model / Run Comparison")
        aggregations = {
            "images": ("image", "count"),
            "expected": ("expected_total", "sum"),
            "missed": ("missed_total", "sum"),
            "false_positives": ("false_positive_total", "sum"),
            "low_confidence": ("low_confidence_total", "sum"),
            "mean_precision": ("precision", "mean"),
            "mean_recall": ("recall", "mean"),
        }
        optional_aggregations = {
            "mean_iou": ("mean_iou", "mean"),
            "map50": ("map50", "mean"),
            "map50_95": ("map50_95", "mean"),
        }
        aggregations.update(
            {
                output_column: aggregation
                for output_column, aggregation in optional_aggregations.items()
                if aggregation[0] in summary_df
            }
        )
        comparison = summary_df.groupby("run_id", as_index=False).agg(**aggregations)
        sort_column = "map50" if "map50" in comparison else "mean_recall"
        comparison = comparison.sort_values(sort_column, ascending=False)
        detection_counts = detections_df.groupby("run_id").size().rename("detections")
        comparison = comparison.join(detection_counts, on="run_id")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("#### Safety Lens")
    if run_summary.empty:
        st.info("No run summary is available for safety interpretation.")
    else:
        chosen_run = st.selectbox(
            "Safety lens run",
            run_summary["run_id"].tolist(),
            index=0,
        )
        selected_run_row = run_summary[run_summary["run_id"] == chosen_run].iloc[0].to_dict()
        selected_run_rows = summary_df[summary_df["run_id"] == chosen_run]
        selected_run_detections = detections_df[detections_df["run_id"] == chosen_run]
        expected_objects = aggregate_count_dict(selected_run_rows["expected_counts"])
        display_threshold = (
            float(selected_run_detections["confidence_threshold"].iloc[0])
            if "confidence_threshold" in selected_run_detections and not selected_run_detections.empty
            else 0.25
        )
        lens_result = evaluate_safety_lens(
            raw_detections=detections_from_dataframe(selected_run_detections),
            display_detections=detections_from_dataframe(selected_run_detections),
            expected_objects=expected_objects,
            display_threshold=display_threshold,
            low_conf_threshold=display_threshold,
            scenario_tags=infer_scenario_tags(chosen_run),
            metrics={
                "precision": float(selected_run_row["mean_precision"]),
                "recall": float(selected_run_row["mean_recall"]),
                "display_threshold": display_threshold,
                "low_confidence_threshold": display_threshold,
                "map50": None if pd.isna(selected_run_row.get("map50")) else float(selected_run_row["map50"]),
                "map50_95": None if pd.isna(selected_run_row.get("map50_95")) else float(selected_run_row["map50_95"]),
                "ground_truth_boxes_available": bool(selected_run_row.get("ground_truth_boxes", 0)),
            },
        )
        st.markdown(generate_safety_report(lens_result, scenario_name=chosen_run))
        standards_context = load_project1_standard_context(max_chars_per_doc=700)
        with st.expander("Project 1 standards excerpts"):
            st.markdown(build_project1_standards_section(standards_context))

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Detections By Class")
        class_counts = detections_df["label"].value_counts().reset_index()
        class_counts.columns = ["label", "count"]
        st.bar_chart(class_counts.set_index("label"))
        st.dataframe(class_counts, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Missed Objects By Class")
        missed_counts = aggregate_count_column(summary_df["missed_objects"])
        if missed_counts.empty:
            st.success("No missed objects recorded in the batch summary.")
        else:
            st.bar_chart(missed_counts.set_index("label"))
            st.dataframe(missed_counts, use_container_width=True, hide_index=True)

    st.markdown("#### Lowest Recall Images")
    worst_cases = summary_df.sort_values(["recall", "missed_total"], ascending=[True, False]).head(20)
    st.dataframe(
        worst_cases[
            [
                "image",
                "detected_total",
                "expected_total",
                "missed_total",
                "false_positive_total",
                "low_confidence_total",
                "precision",
                "recall",
                *[column for column in ["mean_iou", "map50", "map50_95"] if column in worst_cases],
                "missed_objects",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Low Confidence Examples")
    low_conf = detections_df[detections_df["low_confidence"] == True].copy()
    if low_conf.empty:
        st.success("No low-confidence detections recorded.")
    else:
        st.dataframe(
            low_conf.sort_values("confidence").head(50),
            use_container_width=True,
            hide_index=True,
        )

    report_paths = sorted(OUTPUTS_DIR.glob(BATCH_REPORT_GLOB))
    if report_paths:
        st.markdown("#### Colab Batch Report")
        selected_report = st.selectbox(
            "Report file",
            report_paths,
            index=report_paths.index(pick_latest_path(report_paths)) if pick_latest_path(report_paths) in report_paths else 0,
            format_func=lambda path: path.name,
        )
        st.markdown(selected_report.read_text(encoding="utf-8"))


st.title("Perception Safety Evaluation Copilot")
st.caption("MVP for image-based object detection, perception failure analysis, and safety-focused reporting.")

app_mode = st.sidebar.radio(
    "App mode",
    ["Single Image Evaluation", "Batch Results Dashboard"],
)

if app_mode == "Batch Results Dashboard":
    render_batch_dashboard()
    st.stop()

with st.sidebar:
    st.header("Evaluation Setup")
    model_name = st.selectbox(
        "YOLO model",
        ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"],
        index=0,
    )
    confidence_threshold = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)
    low_confidence_threshold = st.slider("Low-confidence safety threshold", 0.10, 0.95, 0.50, 0.05)
    expected_objects_text = st.text_area(
        "Expected objects",
        value="person: 1\ncar: 2\ntraffic light: 1",
        height=120,
        help="Examples: person: 1, car: 2, bicycle: 1. This drives safety severity even when objects disappear above the display threshold.",
    )
    st.caption(
        f"Safety lens always keeps raw detections above the internal candidate threshold {INTERNAL_CANDIDATE_THRESHOLD:.2f}. "
        "The display threshold only controls what is shown as normal detections."
    )
    st.divider()
    st.header("Project 1 Bridge")
    include_project1_context = st.checkbox("Add Project 1 safety context", value=True)
    project1_profile_path = st.text_input(
        "Project 1 nuScenes profile",
        value=str(DEFAULT_NUSCENES_PROFILE),
        help="Used to enrich the safety report. Later this can be replaced by a live MCP call.",
    )
    st.divider()
    st.subheader("Saved Evaluations")
    render_recent_history()

scenario_name = st.text_input(
    "Scenario name",
    value="Urban driving scene with vulnerable road users",
    placeholder="Example: Night urban AEB pedestrian crossing",
)

image_source = st.radio("Image source", ["Upload image", "Project 1 / nuScenes sample"], horizontal=True)

image_rgb = None
image_name = ""
nuscenes_expected_counts: dict[str, int] = {}
nuscenes_ground_truth_boxes = []
nuscenes_context = ""

if image_source == "Upload image":
    uploaded_file = st.file_uploader(
        "Upload a driving image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image)
        image_name = uploaded_file.name
else:
    default_nuscenes_metadata_root = DEFAULT_PROJECT1_DIR / "datasets" / "nuscenes" / "v1.0-trainval"
    default_nuscenes_data_root = DEFAULT_PROJECT1_DIR / "datasets" / "nuscenes"
    nuscenes_root = st.text_input(
        "nuScenes metadata root (`v1.0-trainval`)",
        value=str(default_nuscenes_metadata_root),
        help="Folder containing sample.json, sample_data.json, sample_annotation.json, and other metadata files.",
    )
    nuscenes_data_root = st.text_input(
        "nuScenes data root",
        value=str(default_nuscenes_data_root),
        help="Folder containing samples/CAM_* images. Usually the parent of v1.0-trainval. Leave as-is if images are not available.",
    )
    st.caption(
        "If Project 1 only has the `v1.0-trainval` metadata folder, Project 3 can use the safety profile "
        "and annotations, but image loading needs the matching `samples/CAM_*` files."
    )
    channel = st.selectbox(
        "Camera channel",
        ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
        index=0,
    )

    root_path = Path(nuscenes_root)
    if not is_nuscenes_root(root_path):
        st.warning(
            "No complete nuScenes metadata root found at this path. "
            "Point this to the `v1.0-trainval` folder that contains the nuScenes JSON files."
        )
    else:
        with st.spinner("Scanning nuScenes camera samples..."):
            samples = get_nuscenes_samples(nuscenes_root, nuscenes_data_root, channel, limit=50)
        if not samples:
            st.warning(
                f"Metadata was found, but no existing key-frame images were found for {channel}. "
                "This usually means only the `v1.0-trainval` metadata folder is present and the `samples/` image folder is missing."
            )
        else:
            selected = st.selectbox("nuScenes sample", samples, format_func=lambda item: item.label)
            image = Image.open(selected.image_path).convert("RGB")
            image_rgb = np.array(image)
            image_name = selected.image_path.name
            nuscenes_context = " | ".join(
                part
                for part in [selected.scene_name, selected.scene_description, selected.channel]
                if part
            )
            with st.spinner("Loading nuScenes annotations for expected counts and boxes..."):
                nuscenes_expected_counts = get_nuscenes_expected_counts(nuscenes_root, selected.sample_token)
                nuscenes_ground_truth_boxes = get_nuscenes_ground_truth_boxes(
                    nuscenes_root,
                    selected.sample_token,
                    selected.sample_data_token,
                    image_rgb.shape[1],
                    image_rgb.shape[0],
                )
            if nuscenes_context:
                st.caption(nuscenes_context)

if nuscenes_expected_counts:
    expected_objects_text = counts_to_ground_truth_text(nuscenes_expected_counts)

if image_rgb is None:
    st.info("Upload a driving image or select a nuScenes sample to run the perception safety evaluation.")
    st.stop()

st.subheader("Input Preview")
st.image(image_rgb, width="stretch")

try:
    model = get_model(model_name)
    raw_detections = run_yolo_detection(
        model,
        image_rgb,
        INTERNAL_CANDIDATE_THRESHOLD,
        image_id=image_name,
        model_name=model_name,
    )
except ModuleNotFoundError as exc:
    if exc.name == "ultralytics":
        st.error(
            "YOLO inference requires the `ultralytics` package. "
            "Install the project requirements in the same Python environment used to run Streamlit."
        )
        st.code("pip install -r requirements.txt", language="bash")
    else:
        st.error(f"Missing Python package: {exc.name}")
    st.stop()
except Exception as exc:
    st.error(f"YOLO inference failed: {exc}")
    st.stop()

display_detections = [detection for detection in raw_detections if detection.confidence >= confidence_threshold]
expected_counts = parse_expected_objects(expected_objects_text)
evaluation = evaluate_detections(
    display_detections,
    expected_counts=expected_counts,
    low_confidence_threshold=low_confidence_threshold,
    ground_truth_boxes=nuscenes_ground_truth_boxes,
    iou_threshold=0.5,
)
records = detections_to_records(display_detections)
annotated = draw_detections(image_rgb, display_detections)
metrics = metrics_to_dict(evaluation)
metrics["display_threshold"] = confidence_threshold
metrics["low_confidence_threshold"] = low_confidence_threshold
metrics["ground_truth_boxes_available"] = bool(nuscenes_ground_truth_boxes)
scenario_tags = infer_scenario_tags(" ".join(part for part in [scenario_name, nuscenes_context] if part))
report = generate_markdown_report(
    scenario_name=scenario_name,
    image_name=image_name,
    detections=display_detections,
    result=evaluation,
    confidence_threshold=confidence_threshold,
    low_confidence_threshold=low_confidence_threshold,
)
safety_lens_result = evaluate_safety_lens(
    raw_detections=raw_detections,
    display_detections=display_detections,
    expected_objects=expected_counts,
    display_threshold=confidence_threshold,
    low_conf_threshold=low_confidence_threshold,
    scenario_tags=scenario_tags,
    metrics=metrics,
)
report = report + "\n\n" + generate_safety_report(
    safety_lens_result,
    scenario_name=scenario_name or nuscenes_context,
    metrics=metrics,
)
if include_project1_context:
    project1_context = load_nuscenes_safety_profile(Path(project1_profile_path))
    standards_context = load_project1_standard_context()
    context_query = nuscenes_context or scenario_name or "perception dataset coverage and safety evaluation"
    report = (
        report
        + "\n\n"
        + build_project1_standards_section(standards_context)
        + "\n\n"
        + build_project1_context_section(project1_context, context_query)
    )

result_left, result_center = st.columns(2)
with result_left:
    st.subheader("Original Image")
    st.image(image_rgb, width="stretch")

with result_center:
    st.subheader("Detected Objects")
    st.image(annotated, width="stretch")

st.subheader("Safety Lens")
st.markdown(
    generate_safety_report(
        safety_lens_result,
        scenario_name=scenario_name or nuscenes_context,
        metrics=metrics,
    )
)

metric_cols = st.columns(7)
metric_cols[0].metric("Detected", metrics["detected_total"])
metric_cols[1].metric("Missed", metrics["missed_total"])
metric_cols[2].metric("False Positives", metrics["false_positive_total"])
metric_cols[3].metric("Low Confidence", metrics["low_confidence_total"])
metric_cols[4].metric(
    "Recall",
    "N/A" if metrics["recall"] is None else f"{metrics['recall']:.2f}",
)
metric_cols[5].metric("mAP50", "N/A" if metrics["map50"] is None else f"{metrics['map50']:.2f}")
metric_cols[6].metric("mAP50-95", "N/A" if metrics["map50_95"] is None else f"{metrics['map50_95']:.2f}")

st.subheader("Detection Results")
render_detection_table(records)

st.subheader("Safety Evaluation")
summary = pd.DataFrame(
    [
        {"category": "Detected objects", "value": metrics["detected_counts"] or {}},
        {"category": "Expected objects", "value": metrics["expected_counts"] or {}},
        {"category": "Missed objects", "value": metrics["missed_objects"] or {}},
        {"category": "False positives", "value": metrics["false_positives"] or {}},
    ]
)
st.dataframe(summary, use_container_width=True, hide_index=True)

save_button = st.button("Save Evaluation Snapshot")
evaluation_id = None
if save_button:
    evaluation_id = save_evaluation(
        db_path=DEFAULT_DB_PATH,
        scenario_name=scenario_name,
        image_name=image_name,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        low_confidence_threshold=low_confidence_threshold,
        detections=records,
        expected=expected_counts,
        metrics=metrics,
        report_markdown=report,
    )
    st.success(f"Evaluation saved to {Path(DEFAULT_DB_PATH)} with ID {evaluation_id}.")

st.subheader("Generated Safety Report")
st.markdown(report)
st.download_button(
    "Download Report",
    data=report,
    file_name=f"perception_safety_report_{image_name or 'scene'}.md",
    mime="text/markdown",
)
