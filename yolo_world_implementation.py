"""
YOLO-World Video Annotation Script  v1
Classes : cat, dog
Input   : a single video file
Outputs :
  - annotated_frames/   → annotated JPG frames  (optional)
  - labels/             → YOLO-format .txt label per frame
  - detections.json     → full detection results
  - annotated_video.mp4 → (optional) re-encoded annotated video

Model variants:
  1 -> Small  (yolov8s-worldv2.pt)   fastest
  2 -> Medium (yolov8m-worldv2.pt)
  3 -> Large  (yolov8l-worldv2.pt)
  4 -> X      (yolov8x-worldv2.pt)   most accurate

Requirements:
    pip install ultralytics opencv-python tqdm
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLASSES        = ["cat", "dog"]
DISPLAY_LABELS = {"cat": "cat", "dog": "dog"}
CONFIDENCE     = 0.25
IOU_THRESHOLD  = 0.45

MODEL_OPTIONS = {
    "1": ("yolov8s-worldv2.pt", "Small  — fastest, least accurate"),
    "2": ("yolov8m-worldv2.pt", "Medium"),
    "3": ("yolov8l-worldv2.pt", "Large"),
    "4": ("yolov8x-worldv2.pt", "X      — slowest, most accurate"),
}

CLASS_COLORS = {
    "cat": ( 57, 255, 132),   # green
    "dog": ( 55, 162, 255),   # blue
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
INPUT_VIDEO = r"VIDEO_PATH"
OUTPUT_DIR  = r"OUTPUT_PATH"

# ─────────────────────────────────────────────────────────────────────────────
# SPACE-SAVING OPTIONS
# ─────────────────────────────────────────────────────────────────────────────
SAVE_ANNOTATED_FRAMES = True
SAVE_ANNOTATED_VIDEO  = True
SAVE_RESOLUTION       = (1920, 1080)   # None = keep original resolution
JPEG_QUALITY          = 85

# ─────────────────────────────────────────────────────────────────────────────
# FRAME SKIP
# 1=every frame, 2=every 2nd, 10=every 10th, etc.
# NOTE: Must be >= 1. Setting to 0 will raise a ZeroDivisionError.
# ─────────────────────────────────────────────────────────────────────────────
FRAME_SKIP = 1  # BUG FIX: was 0 in original; 0 causes ZeroDivisionError

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# CHUNK_SIZE = 0  → process all frames in one run (no chunking)
# CHUNK_SIZE = N  → process N source frames then stop
#
# AUTO-RESUME: set RESUME_FROM_FRAME = -1 to automatically continue from
# wherever the last run stopped (reads detections.json).
# Set to 0 to always start fresh from the beginning.
# ─────────────────────────────────────────────────────────────────────────────
RESUME_FROM_FRAME = 0   # -1 = auto-detect from last run
CHUNK_SIZE        = 0   # 0 = no limit, process everything


# ─────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────
def select_model() -> str:
    print("\n" + "=" * 62)
    print("  YOLO-World v2  —  Select Model Variant")
    print("=" * 62)
    for key, (filename, desc) in MODEL_OPTIONS.items():
        print(f"  [{key}]  {desc}")
    print("=" * 62)
    while True:
        choice = input("  Enter choice (1-4): ").strip()
        if choice in MODEL_OPTIONS:
            model_id, desc = MODEL_OPTIONS[choice]
            print(f"\n[INFO] Selected : {desc}")
            print(f"[INFO] Model    : {model_id}\n")
            return model_id
        print("  [!] Invalid — enter a number between 1 and 4.")


def load_model(model_id: str):
    from ultralytics import YOLOWorld

    print(f"[INFO] Loading {model_id} ...")
    model = YOLOWorld(model_id)
    model.set_classes(CLASSES)
    print(f"[INFO] Classes set: {CLASSES}")
    return model


# ─────────────────────────────────────────────
# AUTO-RESUME HELPER
# ─────────────────────────────────────────────
def detect_resume_frame(json_path: Path, frame_skip: int) -> int:
    """Read detections.json and return the next frame to process."""
    if not json_path.exists():
        return 0
    try:
        with open(json_path) as f:
            data = json.load(f)
        if not data:
            return 0
        last_frame = data[-1]["frame"]
        next_frame = last_frame + frame_skip
        print(f"[INFO] Auto-resume: last processed frame = {last_frame}, "
              f"resuming from frame {next_frame}")
        return next_frame
    except Exception as e:
        print(f"[WARN] Could not read {json_path} for auto-resume: {e}. Starting from 0.")
        return 0


# ─────────────────────────────────────────────
# DRAW
# ─────────────────────────────────────────────
def draw_annotations(image: np.ndarray, detections: list) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
        display = DISPLAY_LABELS.get(det["class_name"], det["class_name"])
        conf    = det["confidence"]
        color   = CLASS_COLORS.get(display, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{display} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────
# SAVE YOLO LABELS
# ─────────────────────────────────────────────
def save_yolo_labels(detections, img_w, img_h, label_path):
    lines = []
    for det in detections:
        cls_id = CLASSES.index(det["class_name"])
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    Path(label_path).write_text("\n".join(lines))


# ─────────────────────────────────────────────
# DISK SPACE + ESTIMATE
# ─────────────────────────────────────────────
def check_disk_space(path: Path, warn_gb: float = 5.0):
    import shutil
    _, _, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    print(f"[INFO] Free disk space: {free_gb:.1f} GB")
    if free_gb < warn_gb:
        print(f"[WARN] Only {free_gb:.1f} GB free!")
    return free_gb


def estimate_space(total_frames, img_w, img_h, fps,
                   save_frames, save_video, save_res, jpeg_q,
                   resume_from, chunk_size, frame_skip):
    # BUG FIX: guard against frame_skip < 1 to avoid ZeroDivisionError
    frame_skip = max(1, frame_skip)

    frames_in_range = total_frames - resume_from
    if chunk_size > 0:
        frames_in_range = min(frames_in_range, chunk_size)
    frames_to_process = max(1, (frames_in_range + frame_skip - 1) // frame_skip)

    label_mb = frames_to_process * 3 / 1024
    json_mb  = frames_to_process * 0.5 / 1024

    if save_frames:
        w, h     = save_res if save_res else (img_w, img_h)
        base_kb  = (w * h) / (3840 * 2160) * 400
        q_factor = (jpeg_q / 85) ** 1.5
        frame_mb = frames_to_process * base_kb * q_factor / 1024
    else:
        frame_mb = 0

    if save_video:
        w, h          = save_res if save_res else (img_w, img_h)
        bitrate_mbps  = 5 if h >= 2000 else (2 if h >= 1000 else 1)
        effective_fps = fps / frame_skip
        duration_s    = frames_to_process / effective_fps
        video_mb      = bitrate_mbps * duration_s / 8
    else:
        video_mb = 0

    total_mb = label_mb + json_mb + frame_mb + video_mb
    print(f"\n[INFO] Estimated storage (~{frames_to_process:,} frames after skip={frame_skip}):")
    print(f"         Labels  : {label_mb:>8.0f} MB")
    print(f"         JSON    : {json_mb:>8.0f} MB")
    print(f"         Frames  : {frame_mb:>8.0f} MB  (SAVE_ANNOTATED_FRAMES={save_frames})")
    print(f"         Video   : {video_mb:>8.0f} MB  (SAVE_ANNOTATED_VIDEO={save_video})")
    print(f"         TOTAL ~ : {total_mb/1024:>7.1f} GB\n")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run(video_path, output_dir, model_id, conf, iou,
        save_frames, save_video, save_res, jpeg_quality,
        resume_from, chunk_size, frame_skip):

    # BUG FIX: guard against frame_skip < 1 to avoid ZeroDivisionError
    frame_skip = max(1, frame_skip)

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {video_path.suffix}")

    model = load_model(model_id)

    out        = Path(output_dir)
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = None
    if save_frames:
        frames_dir = out / "annotated_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    cap          = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    img_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── AUTO-RESUME ──────────────────────────────────────────────────────────
    json_path = out / "detections.json"
    if resume_from == -1:
        resume_from = detect_resume_frame(json_path, frame_skip)
    # ─────────────────────────────────────────────────────────────────────────

    check_disk_space(out)
    estimate_space(total_frames, img_w, img_h, fps,
                   save_frames, save_video, save_res, jpeg_quality,
                   resume_from, chunk_size, frame_skip)

    save_w, save_h = save_res if save_res else (img_w, img_h)
    effective_fps  = max(1.0, fps / frame_skip)

    stop_at = total_frames
    if chunk_size > 0:
        stop_at = min(total_frames, resume_from + chunk_size)

    frame_indices = list(range(resume_from, stop_at, frame_skip))

    print(f"[INFO] Video        : {video_path.name}")
    print(f"[INFO] Total frames : {total_frames}  |  FPS: {fps:.1f}  |  {img_w}x{img_h}")
    print(f"[INFO] Save res     : {save_w}x{save_h}  |  JPEG quality: {jpeg_quality}")
    print(f"[INFO] Frame skip   : {frame_skip}  →  {len(frame_indices):,} frames to process")
    print(f"[INFO] Output FPS   : {effective_fps:.1f}  (= {fps:.0f} / {frame_skip})")
    print(f"[INFO] Resume from  : frame {resume_from}")
    print(f"[INFO] Stop at      : frame {stop_at}  ({'no limit' if chunk_size == 0 else f'chunk={chunk_size}'})")
    print(f"[INFO] Output       : {out}\n")

    if resume_from >= total_frames:
        print("[INFO] Nothing to do — video already fully processed.")
        cap.release()
        return

    writer = None
    if save_video:
        out_video_path = out / f"annotated_{video_path.stem}_f{resume_from}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, effective_fps, (save_w, save_h))

    if resume_from > 0 and json_path.exists():
        with open(json_path) as f:
            all_results = json.load(f)
        print(f"[INFO] Loaded {len(all_results)} existing JSON entries.")
    else:
        all_results = []

    frames_done  = 0
    class_counts = {c: 0 for c in CLASSES}
    pbar         = tqdm(total=len(frame_indices), desc="Processing", unit="frame")

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_idx:06d}"

        results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)[0]

        detections = []
        if results.boxes is not None and len(results.boxes):
            boxes   = results.boxes.xyxy.cpu().numpy()
            confs   = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, c, cid in zip(boxes, confs, cls_ids):
                cls_name = CLASSES[cid] if cid < len(CLASSES) else f"class_{cid}"
                detections.append({
                    "class_id":   int(cid),
                    "class_name": cls_name,
                    "confidence": float(round(c, 4)),
                    "bbox_xyxy":  [float(v) for v in box],
                })
                if cls_name in class_counts:
                    class_counts[cls_name] += 1

        save_yolo_labels(detections, img_w, img_h, labels_dir / f"{frame_name}.txt")

        if save_frames or save_video:
            annotated = draw_annotations(frame, detections)
            if (save_w, save_h) != (img_w, img_h):
                annotated = cv2.resize(annotated, (save_w, save_h),
                                       interpolation=cv2.INTER_AREA)
            if save_frames and frames_dir is not None:
                cv2.imwrite(str(frames_dir / f"{frame_name}.jpg"),
                            annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if writer is not None:
                writer.write(annotated)

        all_results.append({
            "frame":      frame_idx,
            "frame_name": frame_name,
            "width":      img_w,
            "height":     img_h,
            "detections": detections,
        })

        frames_done += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    last_frame = frame_indices[frames_done - 1] if frames_done > 0 else resume_from
    next_frame = last_frame + frame_skip

    print("\n" + "=" * 55)
    print(f"  Model            : {model_id}")
    print(f"  Video            : {video_path.name}")
    print(f"  Frame skip       : {frame_skip}")
    print(f"  Frames processed : {frames_done}  (idx {resume_from} → {last_frame})")
    print(f"  Total detections : {sum(class_counts.values())}")
    print("  Per-class        :")
    for cls, cnt in class_counts.items():
        print(f"    {cls:<10} {cnt}")
    print(f"\n  YOLO labels      → {labels_dir}")
    print(f"  JSON             → {json_path}")
    if save_frames:
        print(f"  Annotated frames → {frames_dir}")
    if writer:
        print(f"  Annotated video  → {out / f'annotated_{video_path.stem}_f{resume_from}.mp4'}")

    if next_frame < total_frames:
        print(f"\n  [!] Chunk complete.")
        print(f"      Next run will auto-resume from frame {next_frame}  (RESUME_FROM_FRAME = -1)")
    else:
        print(f"\n  [✓] All {total_frames} frames fully processed.")
    print("=" * 55)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    selected_model = select_model()
    run(
        video_path   = INPUT_VIDEO,
        output_dir   = OUTPUT_DIR,
        model_id     = selected_model,
        conf         = CONFIDENCE,
        iou          = IOU_THRESHOLD,
        save_frames  = SAVE_ANNOTATED_FRAMES,
        save_video   = SAVE_ANNOTATED_VIDEO,
        save_res     = SAVE_RESOLUTION,
        jpeg_quality = JPEG_QUALITY,
        resume_from  = RESUME_FROM_FRAME,
        chunk_size   = CHUNK_SIZE,
        frame_skip   = FRAME_SKIP,
    )