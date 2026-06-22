# Perception Safety Evaluation Copilot

Project 3 portfolio MVP: a computer-vision evaluation tool that connects perception AI results with safety engineering reasoning.

## Highlight Screenshots

![Safety Lens 0](assets/screenshots/Safety_Lens_0.png)
*Hero view of the Perception Safety Evaluation Copilot, combining perception outputs with safety-focused analysis.*

![Safety Lens 1](assets/screenshots/Safety_Lens_1.png)
*Single-image evaluation with the original scene, detected objects, and the Safety Lens workflow.*

![Safety Lens 2](assets/screenshots/Safety_Lens_2.png)
*Safety Lens v2 translating perception failures into standards-aware safety interpretation.*

![Safety Lens 3](assets/screenshots/Safety_Lens_3.png)
*Detailed safety output connecting observed perception gaps to ISO 21448 / SOTIF, ISO 8800, and ISO 26262 reasoning.*

![Safety Lens 4](assets/screenshots/Safety_Lens_4.png)
*Portfolio-ready view of the perception safety evaluation experience and report structure.*

This first version focuses on a small, runnable workflow:

- Upload a driving image.
- Or select a local nuScenes camera sample from Project 1's dataset folder if available.
- Run object detection with Ultralytics YOLO.
- Display detections with bounding boxes and confidence scores.
- Optionally enter expected object counts as simple ground truth.
- Calculate detected objects, missed expected objects, false positives, low-confidence detections, precision, and recall.
- Save each evaluation to local SQLite.
- Generate a safety-oriented perception failure report.

## Model Evaluation Summary

The project now includes a fine-tuned `YOLO11s` disturbance-aware detector trained on a YOLO-formatted BDD100K driving-scene dataset.

Headline results:

- Precision: `~0.72`
- Recall: `~0.46`
- mAP50: `~0.51`
- mAP50-95: `~0.28`
- Best F1 confidence operating point: `~0.256`

Engineering interpretation:

- the model converged stably across 20 epochs
- the dominant failure mode is missed detections rather than class confusion
- vehicle and infrastructure classes perform better than vulnerable road user classes
- pedestrians, riders, bicycles, and motorcycles remain the most safety-relevant weakness area
- for safety analysis, threshold increases above `0.25` should be treated carefully because recall drops quickly

This result directly reinforces the purpose of Project 3: the tool should not stop at generic mAP reporting. It should surface missed-object evidence, vulnerable-road-user risk, threshold sensitivity, and standards-oriented safety implications.

Full write-up: [docs/yolo11s_finetuning_summary.md](/Users/chongharnzhin/Documents/Personal/AI/Bootcamp/00_W8-W9_Final_Project/Perception_Safety_Evaluation_Copilot/docs/yolo11s_finetuning_summary.md)

## Before vs Fine-Tuned Model

Base `YOLO11s` result before disturbance-focused fine-tuning:

![YOLO11s Before Fine-Tuning](assets/screenshots/yolo11s.png)

Fine-tuned `YOLO11s Disturbance Fine-Tuned` result after BDD100K-based training:

![YOLO11s After Fine-Tuning](assets/screenshots/yolo11s_fine_tuned.png)

This comparison helps show the practical purpose of Project 3: not just running perception models, but evaluating whether fine-tuning improves safety-relevant detection behavior under disturbance conditions.

## Why This Complements Projects 1 and 2

Project 1, `Autonomous_Driving_Safety_Analyst`, provides standards and safety context through an LLM/RAG and MCP-based knowledge service.

Project 2, `Agentic_Document_AI_Platform_for_Safety_Engineering`, provides requirements, traceability, test-case generation, monitoring, workflow tracking, and MLflow-style evaluation.

Project 3 adds perception model evaluation: it turns image-level detection behavior into safety-relevant evidence that can later connect to standards, requirements, test cases, and traceability.

## Folder Structure

```text
Perception_Safety_Evaluation_Copilot/
  app.py
  requirements.txt
  README.md
  assets/
    screenshots/
  data/
  src/
    perception_safety_copilot/
      detection.py
      evaluation.py
      nuscenes_connector.py
      project1_bridge.py
      reporting.py
      safety_lens.py
      storage.py
  tests/
    test_evaluation.py
    test_safety_lens.py
```

## Local Setup

From this folder:

```bash
cd Perception_Safety_Evaluation_Copilot
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The first YOLO run may download model weights such as `yolov8n.pt` or
`yolo11s.pt`.

Run tests:

```bash
pytest
```

## YOLO Fine-Tuning Setup

You now have a ready-to-use local training config for the complete YOLO-formatted
BDD100K dataset:

- Dataset YAML: [training/bdd100k_yolo_local.yaml](/Users/chongharnzhin/Documents/Personal/AI/Bootcamp/00_W8-W9_Final_Project/Perception_Safety_Evaluation_Copilot/training/bdd100k_yolo_local.yaml)
- Training helper: [scripts/train_yolo_bdd100k.py](/Users/chongharnzhin/Documents/Personal/AI/Bootcamp/00_W8-W9_Final_Project/Perception_Safety_Evaluation_Copilot/scripts/train_yolo_bdd100k.py)

Smoke test locally on your Mac first:

```bash
cd Perception_Safety_Evaluation_Copilot
source .venv/bin/activate
python scripts/train_yolo_bdd100k.py --model yolo11s.pt --epochs 1 --batch 8 --device mps --name smoke_yolo11s
```

If that works, step up to a longer run:

```bash
python scripts/train_yolo_bdd100k.py --model yolo11s.pt --epochs 20 --batch 16 --device mps --name yolo11s_disturbance_ft
```

For Colab or Linux GPU, switch `--device` to `0`.

Training outputs will land under:

```text
runs/bdd100k_training/
```

The old raw BDD100K folder `archive (1)` is no longer required for this
fine-tuning path. Keep it only if you still want to generate custom subsets from
the original JSON metadata using `scripts/create_bdd100k_yolo_subset.py`.

## Ground Truth Input Format

Ground truth is optional in the MVP. Enter one expected object class per line:

```text
person: 1
car: 2
traffic light: 1
```

Also accepted:

```text
person,1
car
```

If ground truth is provided, the app calculates count-based precision and recall. This is intentionally simple for the MVP. A later version should support bounding-box labels and IoU-based matching.

## Project 1 and nuScenes Connection

The app has an optional Project 1 bridge:

- `Add Project 1 safety context` appends Project 1's `standards_pdfs/nuscenes_dataset_profile.md` to the generated safety report.
- `Project 1 / nuScenes sample` can load camera key frames from a local nuScenes root and convert nuScenes annotations into expected object counts.

Expected nuScenes root shape:

```text
Autonomous_Driving_Safety_Analyst/
  datasets/
    nuscenes/
      samples/
        CAM_FRONT/
          *.jpg
      v1.0-trainval/
        sample.json
        sample_data.json
        sample_annotation.json
        sensor.json
        calibrated_sensor.json
        scene.json
        instance.json
        category.json
```

In the app, set `nuScenes metadata root` to the `v1.0-trainval` folder. Set
`nuScenes data root` to the parent folder that contains `samples/`.

If Project 1 only contains `v1.0-trainval`, then Project 3 can still use the
metadata-derived Project 1 safety profile, but it cannot display camera images
until the matching `samples/CAM_*` files are available.

If Google Drive space is limited, create a small subset locally instead of
uploading the full dataset:

```bash
python scripts/create_nuscenes_colab_subset.py \
  --metadata-root /path/to/nuscenes/v1.0-trainval \
  --data-root /path/to/nuscenes \
  --output-root ~/Desktop/nuscenes_subset \
  --channel CAM_FRONT \
  --limit 200
```

Then upload only `~/Desktop/nuscenes_subset` or a zip of that folder to Colab or
Google Drive.

Project 1's current MCP server is still the right long-term interface for standards retrieval. This MVP uses the local Project 1 nuScenes profile as a lightweight bridge first, so Project 3 remains runnable even when the MCP server is not active. The next integration step is to replace that local profile read with a live MCP call to `search_combined_safety_context`.

## Current MVP Capabilities

- Image upload for JPG, JPEG, and PNG.
- Optional local nuScenes image sample loading.
- Optional Project 1 nuScenes safety profile context in generated reports.
- Batch results dashboard for Colab-generated CSV and Markdown outputs.
- YOLO model selection with YOLOv8 and YOLO11 options, including `yolo11s.pt`
  for stronger portfolio experiments.
- Confidence threshold controls.
- Low-confidence threshold controls for safety review.
- Detection overlay using OpenCV.
- Detection table with labels, confidence scores, and bounding boxes.
- SQLite persistence in `data/perception_evaluations.sqlite3`.
- Recent evaluation history in the sidebar.
- Downloadable Markdown safety report.

## Batch Evaluation Dashboard

After running the Colab notebook, place these files in `outputs/`:

```text
outputs/
  perception_yolo_results.csv
  perception_eval_summary.csv
  perception_failure_report.md
```

For model comparisons, prefer model-specific output names:

```text
outputs/
  perception_yolo_results_yolov8n_conf0p25.csv
  perception_eval_summary_yolov8n_conf0p25.csv
  perception_failure_report_yolov8n_conf0p25.md
  perception_yolo_results_yolo11s_conf0p25.csv
  perception_eval_summary_yolo11s_conf0p25.csv
  perception_failure_report_yolo11s_conf0p25.md
```

Then start the app and choose `Batch Results Dashboard` in the sidebar:

```bash
streamlit run app.py
```

The dashboard reads both the original filenames and model-specific filenames. It
shows aggregate image counts, detections, expected objects, missed objects,
low-confidence detections, mean recall, model/run comparison, class
distributions, lowest recall images, the generated batch report, and a safety
lens section that interprets the run using ISO 26262, ISO 21448 / SOTIF, and
ISO 8800 concepts from Project 1.

## Next Milestones

1. Add video frame extraction.
   - Sample frames from uploaded videos.
   - Run detection per frame.
   - Summarize temporal failure patterns.

2. Add MLflow tracking.
   - Track model name, thresholds, metrics, image metadata, and report artifacts.
   - Compare YOLO model versions and threshold strategies.
   - Recommended first comparison: `yolov8n.pt` versus `yolo11s.pt` at
     confidence thresholds `0.25` and `0.50`.

3. Add Project 1 MCP retrieval.
   - Pull ISO 26262, ISO 21448/SOTIF, and ISO 8800 context.
   - Enrich the generated report with standards-grounded safety rationale.
   - Replace the local profile bridge with live calls to `search_combined_safety_context`.

4. Add Project 2 API integration.
   - Link perception failures to requirements, traceability, and generated test cases.
   - Push evaluation records into project workspaces.

5. Add a FastAPI backend.
   - Move inference, persistence, and reporting into API endpoints.
   - Keep Streamlit as the first UI client.

6. Add a perception model evaluation dashboard.
   - Aggregate performance across scenarios.
   - Track missed objects, low-confidence classes, false positives, and scenario coverage.

## Notes

The MVP uses object class counts for optional ground truth, not bounding-box annotations. Treat this as an early safety triage tool, not a formal model validation system yet.
