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
from src.perception_safety_copilot.evaluation import evaluate_detections, parse_ground_truth
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
    load_nuscenes_safety_profile,
)
from src.perception_safety_copilot.reporting import generate_markdown_report
from src.perception_safety_copilot.storage import (
    DEFAULT_DB_PATH,
    load_recent_evaluations,
    save_evaluation,
)


OUTPUTS_DIR = Path("outputs")
BATCH_DETECTIONS_PATH = OUTPUTS_DIR / "perception_yolo_results.csv"
BATCH_SUMMARY_PATH = OUTPUTS_DIR / "perception_eval_summary.csv"
BATCH_REPORT_PATH = OUTPUTS_DIR / "perception_failure_report.md"


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


def metrics_to_dict(result) -> dict:
    return {
        "detected_total": sum(result.detected_counts.values()),
        "missed_total": sum(result.missed_objects.values()),
        "false_positive_total": sum(result.false_positives.values()),
        "low_confidence_total": len(result.low_confidence_detections),
        "precision": result.precision,
        "recall": result.recall,
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


def render_batch_dashboard() -> None:
    st.subheader("Batch Evaluation Results")
    st.caption("Colab-generated perception evaluation artifacts from `outputs/`.")

    if not BATCH_SUMMARY_PATH.exists() or not BATCH_DETECTIONS_PATH.exists():
        st.info(
            "Batch outputs were not found yet. Run the Colab notebook, then place "
            "`perception_eval_summary.csv` and `perception_yolo_results.csv` in `outputs/`."
        )
        return

    summary_df = pd.read_csv(BATCH_SUMMARY_PATH)
    detections_df = pd.read_csv(BATCH_DETECTIONS_PATH)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Images", f"{len(summary_df):,}")
    metric_cols[1].metric("Detections", f"{len(detections_df):,}")
    metric_cols[2].metric("Expected", f"{int(summary_df['expected_total'].sum()):,}")
    metric_cols[3].metric("Missed", f"{int(summary_df['missed_total'].sum()):,}")
    metric_cols[4].metric("Low Conf.", f"{int(summary_df['low_confidence_total'].sum()):,}")
    metric_cols[5].metric("Mean Recall", f"{summary_df['recall'].mean():.2f}")

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

    if BATCH_REPORT_PATH.exists():
        st.markdown("#### Colab Batch Report")
        st.markdown(BATCH_REPORT_PATH.read_text(encoding="utf-8"))


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
    model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"], index=0)
    confidence_threshold = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)
    low_confidence_threshold = st.slider("Low-confidence safety threshold", 0.10, 0.95, 0.50, 0.05)
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
            with st.spinner("Loading nuScenes annotations for expected object counts..."):
                nuscenes_expected_counts = get_nuscenes_expected_counts(nuscenes_root, selected.sample_token)
            if nuscenes_context:
                st.caption(nuscenes_context)

default_ground_truth = counts_to_ground_truth_text(nuscenes_expected_counts)
ground_truth_text = st.text_area(
    "Optional expected objects",
    value=default_ground_truth,
    placeholder="person: 1\ncar: 2\ntraffic light: 1",
    height=120,
    help="Use one class per line. Examples: 'person: 2', 'car,1', or 'traffic light'.",
)

if image_rgb is None:
    st.info("Upload a driving image or select a nuScenes sample to run the perception safety evaluation.")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.subheader("Input Image")
    st.image(image_rgb, use_column_width=True)

run_button = st.button("Run Perception Evaluation", type="primary")

if not run_button:
    st.stop()

try:
    model = get_model(model_name)
    detections = run_yolo_detection(model, image_rgb, confidence_threshold)
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

expected_counts = parse_ground_truth(ground_truth_text)
evaluation = evaluate_detections(
    detections,
    expected_counts=expected_counts,
    low_confidence_threshold=low_confidence_threshold,
)
records = detections_to_records(detections)
annotated = draw_detections(image_rgb, detections)
metrics = metrics_to_dict(evaluation)
report = generate_markdown_report(
    scenario_name=scenario_name,
    image_name=image_name,
    detections=detections,
    result=evaluation,
    confidence_threshold=confidence_threshold,
    low_confidence_threshold=low_confidence_threshold,
)
if include_project1_context:
    project1_context = load_nuscenes_safety_profile(Path(project1_profile_path))
    context_query = nuscenes_context or scenario_name or "perception dataset coverage and safety evaluation"
    report = (
        report
        + "\n\n"
        + build_project1_context_section(project1_context, context_query)
    )

with right:
    st.subheader("Detection Overlay")
    st.image(annotated, use_column_width=True)

metric_cols = st.columns(5)
metric_cols[0].metric("Detected", metrics["detected_total"])
metric_cols[1].metric("Missed", metrics["missed_total"])
metric_cols[2].metric("False Positives", metrics["false_positive_total"])
metric_cols[3].metric("Low Confidence", metrics["low_confidence_total"])
metric_cols[4].metric(
    "Recall",
    "N/A" if metrics["recall"] is None else f"{metrics['recall']:.2f}",
)

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
    file_name=f"perception_safety_report_{evaluation_id}.md",
    mime="text/markdown",
)
