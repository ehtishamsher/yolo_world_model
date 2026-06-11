# 🐾 YOLO-World Video Annotator

Annotate video files with **cat** and **dog** detections using [YOLO-World v2](https://github.com/AILab-CVC/YOLO-World) — an open-vocabulary, zero-shot object detector. Outputs annotated frames, YOLO-format labels, a JSON results file, and an optional annotated video.

---

## Features

- Zero-shot detection via YOLO-World v2 (no fine-tuning required)
- 4 model sizes: Small → X (speed vs. accuracy trade-off)
- Bounding-box annotations drawn on frames and/or video
- YOLO-format `.txt` label files per frame
- Full `detections.json` with frame-level metadata
- **Chunked processing** — stop and resume at any point
- **Auto-resume** — picks up from the last processed frame automatically
- **Frame skip** — process every Nth frame to save time and disk space
- Disk space estimation before processing begins

---

## Requirements

```bash
pip install ultralytics opencv-python tqdm
```

> Python 3.8+ recommended. Weights are downloaded automatically by `ultralytics` on first use.

---

## Setup

Open `annotate.py` and set the two required paths near the top:

```python
INPUT_VIDEO = r"path/to/your/video.mp4"
OUTPUT_DIR  = r"path/to/output/folder"
```

Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`

---

## Usage

```bash
python annotate.py
```

You will be prompted to select a model variant:

```
==============================================================
  YOLO-World v2  —  Select Model Variant
==============================================================
  [1]  Small  — fastest, least accurate
  [2]  Medium
  [3]  Large
  [4]  X      — slowest, most accurate
==============================================================
  Enter choice (1-4):
```

---

## Configuration

All options are set as constants at the top of the script.

### Core Detection

| Option          | Default          | Description                       |
| --------------- | ---------------- | --------------------------------- |
| `CLASSES`       | `["cat", "dog"]` | Classes for YOLO-World to detect  |
| `CONFIDENCE`    | `0.25`           | Minimum confidence threshold      |
| `IOU_THRESHOLD` | `0.45`           | Non-max suppression IoU threshold |

### Output Options

| Option                  | Default        | Description                                      |
| ----------------------- | -------------- | ------------------------------------------------ |
| `SAVE_ANNOTATED_FRAMES` | `True`         | Save annotated JPG frames to `annotated_frames/` |
| `SAVE_ANNOTATED_VIDEO`  | `True`         | Re-encode and save annotated video               |
| `SAVE_RESOLUTION`       | `(1920, 1080)` | Output resolution; `None` = keep original        |
| `JPEG_QUALITY`          | `85`           | JPEG compression quality (1–100)                 |

### Frame Control

| Option              | Default | Description                                                  |
| ------------------- | ------- | ------------------------------------------------------------ |
| `FRAME_SKIP`        | `1`     | Process every Nth frame (`1` = every frame, `5` = every 5th) |
| `CHUNK_SIZE`        | `0`     | Max frames per run; `0` = no limit (process all)             |
| `RESUME_FROM_FRAME` | `0`     | Start frame index; `-1` = auto-detect from last run          |

---

## Output Structure

```
output_dir/
├── labels/
│   ├── frame_000000.txt   # YOLO-format bounding boxes
│   ├── frame_000001.txt
│   └── ...
├── annotated_frames/      # (if SAVE_ANNOTATED_FRAMES = True)
│   ├── frame_000000.jpg
│   └── ...
├── detections.json        # Full detection results
└── annotated_video.mp4    # (if SAVE_ANNOTATED_VIDEO = True)
```

### `detections.json` format

```json
[
  {
    "frame": 0,
    "frame_name": "frame_000000",
    "width": 1920,
    "height": 1080,
    "detections": [
      {
        "class_id": 0,
        "class_name": "cat",
        "confidence": 0.812,
        "bbox_xyxy": [134.2, 210.5, 489.1, 601.3]
      }
    ]
  }
]
```

### YOLO label format (`.txt`)

One line per detection: `class_id cx cy width height` (all values normalized 0–1).

---

## Chunked Processing & Auto-Resume

For long videos, you can process in chunks and resume automatically:

```python
CHUNK_SIZE        = 5000   # Process 5000 frames then stop
RESUME_FROM_FRAME = -1     # -1 = auto-detect from detections.json
```

Run the script repeatedly — each run picks up exactly where the last one left off.

---

## Extending to Other Classes

YOLO-World is open-vocabulary. To detect other objects, update the `CLASSES` list and optionally the `CLASS_COLORS` dict:

```python
CLASSES = ["cat", "dog", "bird", "fish"]

CLASS_COLORS = {
    "cat":  ( 57, 255, 132),
    "dog":  ( 55, 162, 255),
    "bird": (255, 200,  50),
    "fish": (255,  80, 150),
}
```

---

## Model Comparison

| Variant | File                 | Speed    | Accuracy |
| ------- | -------------------- | -------- | -------- |
| Small   | `yolov8s-worldv2.pt` | ⚡⚡⚡⚡ | ★★☆☆     |
| Medium  | `yolov8m-worldv2.pt` | ⚡⚡⚡☆  | ★★★☆     |
| Large   | `yolov8l-worldv2.pt` | ⚡⚡☆☆   | ★★★★     |
| X       | `yolov8x-worldv2.pt` | ⚡☆☆☆    | ★★★★+    |

---

## License

This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0.
