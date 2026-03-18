"""
Real-time Tube Detection with Tracking
======================================
- Prefers ONNX model (then NCNN, TFLite, PyTorch)
- Cooldown period prevents duplicate counts
- Unique IDs: TUBE_001, TUBE_002, ...
- Only tube classes, confidence > 0.7
- ByteTrack tracking (same tube = one count)
- Optimized for Raspberry Pi
"""

import cv2
import argparse
import sqlite3
import time
import os
import uuid
import datetime
import numpy as np
import torch
from pathlib import Path
from threading import Thread
from ultralytics import YOLO


def _normalize_name(s: str) -> str:
    """Lowercase and strip spaces/specials for stable name comparison."""
    return (
        s.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("(", "")
        .replace(")", "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Threaded Video Capture (Important for performance, especially on Raspberry Pi)
# ─────────────────────────────────────────────────────────────────────────────
class VideoCaptureAsync:
    def __init__(self, source):
        # Adjust for Windows if it's a camera index
        if os.name == "nt" and isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed, self.frame = grabbed, frame
            else:
                # If frame capture fails, don't spin too fast
                time.sleep(0.01)

    def read(self):
        return self.grabbed, self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────
# Database & Logging
# ─────────────────────────────────────────────────────────────────────────
def setup_db(db_path: Path):
    """Create tube_detections table if it does not exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tube_detections (
            tube_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            brand_name TEXT,
            confidence REAL,
            track_id INTEGER
        )
    """)
    conn.commit()
    return conn


def get_next_tube_id(conn: sqlite3.Connection) -> str:
    """Generates sequential IDs: TUBE_001, TUBE_002, ..."""
    cursor = conn.cursor()
    cursor.execute("SELECT tube_id FROM tube_detections ORDER BY tube_id DESC LIMIT 1")
    row = cursor.fetchone()
    if row is None:
        return "TUBE_001"
    try:
        # Extract number from 'TUBE_XXX'
        current_id_str = str(row[0])
        num = int(current_id_str.replace("TUBE_", ""))
        return f"TUBE_{num + 1:03d}"
    except (ValueError, AttributeError):
        return "TUBE_001"


def log_tube(conn: sqlite3.Connection, tube_id: str, brand_name: str, confidence: float, track_id: int):
    """Logs a single detection to the database."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO tube_detections (tube_id, timestamp, brand_name, confidence, track_id)
        VALUES (?, ?, ?, ?, ?)
    """, (tube_id, timestamp, brand_name, round(confidence, 2), track_id))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────
# Model & Class Loading
# ─────────────────────────────────────────────────────────────────────────
def load_valid_classes(base_dir: Path) -> set:
    """Loads valid tube classes from classes.txt or defaults to known ones."""
    classes_file = base_dir / "data" / "dataset(tubes)" / "classes.txt"
    if not classes_file.exists():
        # Fallback to common brands if classes.txt is missing
        fallback = ["valbet", "araldite", "beutiful-n", "dk gel", "silverkant", "cani-maks"]
        return {_normalize_name(c) for c in fallback}

    with open(classes_file, "r", encoding="utf-8") as f:
        return {_normalize_name(line.strip()) for line in f if line.strip()}


# ─────────────────────────────────────────────────────────────────────────
# Main Detection System
# ─────────────────────────────────────────────────────────────────────────
def run_system(
    source_input,
    imgsz=320,
    min_conf=0.7,
    frame_skip=2,
    min_box_area=3000,
    cooldown_sec=5.0,
    headless=False,
    run_name="balanced_run",
    exclude_list=[]
):
    base_dir = Path(__file__).resolve().parent.parent
    # Hardcoded weights path as requested
    weights_dir = Path("/Volumes/Untitled 2/downloads/final-year-main 2/runs/detect/balanced_run/weights")

    # 1. Weights Hierarchy: ONNX > NCNN > TFLite > PyTorch
    onnx_path = weights_dir / "best.onnx"
    ncnn_path = weights_dir / "best_ncnn_model"
    tflite_path = weights_dir / "best_saved_model" / "best_float32.tflite"
    pt_path = weights_dir / "best.pt"

    if onnx_path.exists():
        print(f"🚀 Loading ONNX model from {onnx_path}")
        model = YOLO(str(onnx_path), task="detect")
    elif ncnn_path.exists():
        print(f"🚀 Loading NCNN model from {ncnn_path}")
        model = YOLO(str(ncnn_path), task="detect")
    elif tflite_path.exists():
        print(f"🚀 Loading TFLite model from {tflite_path}")
        model = YOLO(str(tflite_path), task="detect")
    elif pt_path.exists():
        print(f"✅ Loading PyTorch model from {pt_path}")
        model = YOLO(str(pt_path))
    else:
        raise RuntimeError(
            f"No valid weights found in {weights_dir}. "
            f"Expected: best.onnx, best_ncnn_model, best_float32.tflite, or best.pt"
        )

    # Print model classes for debugging
    print(f"📦 Model Classes ({len(model.names)}): {list(model.names.values())}")

    # 2. Config & Alias Map
    normalized_valid = load_valid_classes(base_dir)
    print(f"🔍 Valid classes filter: {normalized_valid}")

    # Explicitly ignore person-like classes and common distractors to avoid false positives
    ignored_classes = {
        "person", "man", "woman", "child", "human", "face", 
        "phone", "cell phone", "mobile", "hand", "finger", 
        "background", "bottle", "handbag", "backpack"
    }

    alias_map = {
        "valbet": "valbet", "val-bet": "valbet", "val bet": "valbet",
        "silverkant": "silverkant", "silver-kant": "silverkant", "skkant": "silverkant",
        "dk_gel": "dk gel", "dkgel": "dk gel",
        "cani-maks": "cani-maks", "canimaks": "cani-maks",
        "beutiful-n": "beutiful-n", "beautiful-n": "beutiful-n",
        "araldite": "araldite", "halobet": "halobet",
    }

    # 3. Database Initialisation
    db_path = base_dir / "data" / "tube_detections.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = setup_db(db_path)

    # 4. Camera Init
    try:
        source_int = int(source_input)
    except ValueError:
        source_int = source_input

    cap = VideoCaptureAsync(source_int) if isinstance(source_int, int) else cv2.VideoCapture(source_input)

    if not cap.isOpened():
        print(f"❌ Camera/Source not available: {source_input}")
        conn.close()
        return

    # 5. Tracking and Persistence Logic
    last_detection_time = 0.0
    logged_track_ids = set()
    persistence_counter = {} # track_id -> frames_seen_count
    tube_count = 0

    print("=" * 50)
    print(f"Tube Detector | Run: {run_name} | Cooldown: {cooldown_sec}s")
    print("Press 'Q' to quit")
    print("=" * 50)

    frame_count = 0
    fps_start = time.time()
    fps_counter = 0
    display_fps = 0.0

    if not headless:
        cv2.namedWindow("ML Brand Detector", cv2.WINDOW_NORMAL)

    # Convert exclude list to set
    exclude_brands = {b.lower() for b in exclude_list}
    if exclude_brands:
        print(f"🚫 Manually Excluding: {exclude_brands}")

    # 6. Main Loop
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                break

            frame_count += 1
            # Skip frames to reduce load (especially on Pi)
            if frame_count % frame_skip != 0:
                if not headless:
                    # Minimal UI update on skipped frames
                    cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("ML Brand Detector", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # Run inference
            with torch.no_grad():
                results = model.track(
                    frame,
                    conf=min_conf,
                    imgsz=imgsz,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )

            best_detection = None
            best_conf = 0.0

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                clss = results[0].boxes.cls.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                track_ids = (
                    results[0].boxes.id.int().cpu().numpy()
                    if results[0].boxes.id is not None
                    else np.full(len(boxes), -1, dtype=int)
                )

                for box, cls_id, conf, track_id in zip(boxes, clss, confs, track_ids):
                    if conf < min_conf:
                        continue

                    raw_name = model.names.get(int(cls_id), "").lower()
                    
                    # 1. Immediate exclusion of known non-tube classes
                    if raw_name in ignored_classes:
                        continue

                    # 2. Normalize and check against valid brands
                    norm = _normalize_name(raw_name)
                    canonical = alias_map.get(norm, raw_name)

                    if _normalize_name(canonical) not in normalized_valid:
                        continue

                    # 3. Manual Exclusion (Argument based)
                    if canonical in exclude_brands or _normalize_name(canonical) in exclude_brands:
                        continue

                    # 4. Aspect Ratio Filter (Tubes are usually elongated)
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    aspect_ratio = w / h if h != 0 else 0
                    
                    # If it's too square (like a face or a patch of background), it's probably a false positive
                    if 0.7 < aspect_ratio < 1.4:
                        continue

                    # 5. Filter by area (Tubes should be prominent)
                    if (w * h) < min_box_area:
                        continue

                    # 6. Stability/Tracking & Persistence Check
                    if track_id < 0:
                        # Display it in the UI as 'unstable' but don't log it
                        if not headless:
                           cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 1)
                           cv2.putText(frame, "unstable", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                        continue

                    # Track persistence (How many frames has this object existed?)
                    persistence_counter[track_id] = persistence_counter.get(track_id, 0) + 1
                    
                    # Must be seen for at least 4 inference frames (approx 8-10 raw frames) to be trusted
                    if persistence_counter[track_id] < 4:
                        if not headless:
                           cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
                           cv2.putText(frame, f"verifying {persistence_counter[track_id]}/4", (int(x1), int(y1)-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        continue

                    # 7. Deduplicate by track_id
                    if track_id in logged_track_ids:
                        continue

                    # 8. Pick highest confidence verified tube
                    if conf > best_conf:
                        best_conf = float(conf)
                        best_detection = (
                            canonical,
                            float(conf),
                            (int(x1), int(y1), int(x2), int(y2)),
                            int(track_id)
                        )

            now = time.time()
            cooldown_elapsed = (now - last_detection_time) >= cooldown_sec

            # Perform Logging
            if best_detection and cooldown_elapsed:
                brand_name, conf, (x1, y1, x2, y2), track_id = best_detection
                tube_id = get_next_tube_id(conn)
                log_tube(conn, tube_id, brand_name, conf, track_id)

                if track_id >= 0:
                    logged_track_ids.add(track_id)

                last_detection_time = now
                tube_count += 1
                print(f"🔔 [{tube_id}] {brand_name} (conf={conf:.2f}, track_id={track_id}) - Total: {tube_count}")

                if not headless:
                    # Highlight successful detection in Green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{tube_id} {brand_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif best_detection and not headless:
                # Indicate detection detected but waiting for cooldown
                _, _, (x1, y1, x2, y2), _ = best_detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(frame, "waiting CD...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Prevent memory bloat on logged_track_ids
            if len(logged_track_ids) > 100:
                logged_track_ids.clear()

            # FPS Calculation
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                display_fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()

            # HUD Display
            if not headless:
                cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Tubes detected: {tube_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if cooldown_elapsed:
                    cv2.putText(frame, "READY", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    rem = cooldown_sec - (now - last_detection_time)
                    cv2.putText(frame, f"Cooldown: {rem:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                cv2.imshow("ML Brand Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif frame_count % 30 == 0:
                print(f"📊 FPS: {display_fps:.1f} | Tubes: {tube_count} | CD: {'Ready' if cooldown_elapsed else 'Wait'}")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        conn.close()
        print(f"\n✅ Session ended. Total tubes processed: {tube_count}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Tube Brand Detection & Tracking")
    parser.add_argument("--source", type=str, default="0", help="Video source (index or path)")
    parser.add_argument("--run", type=str, default="balanced_run", help="Experiment name (folder in runs/detect/)")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds to wait between loggings")
    parser.add_argument("--min-conf", type=float, default=0.7, help="Detection confidence threshold")
    parser.add_argument("--skip", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    parser.add_argument("--min-area", type=int, default=6000, help="Minimum bounding box area")
    parser.add_argument("--exclude", nargs="+", default=["beutiful-n"], help="Brands to ignore (e.g. beutiful-n silverkant)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical window")

    args = parser.parse_args()

    run_system(
        source_input=args.source,
        imgsz=args.imgsz,
        min_conf=args.min_conf,
        frame_skip=args.skip,
        min_box_area=args.min_area,
        cooldown_sec=args.cooldown,
        headless=args.headless,
        run_name=args.run,
        exclude_list=args.exclude
    )
