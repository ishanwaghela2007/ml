import cv2
import argparse
import uuid
import datetime
import sqlite3
import time
from pathlib import Path
from threading import Thread
from ultralytics import YOLO

# ───────────────────────────────────────────────────────────────────
# OpenVINO-optimized inference for Raspberry Pi 4
# Intel OpenVINO now supports ARM CPUs via pip wheels.
# This script is tuned for maximum FPS on Pi with OpenVINO backend.
# ───────────────────────────────────────────────────────────────────

class VideoCaptureAsync:
    """Threaded video capture — keeps frame grabbing off the inference thread."""
    def __init__(self, source):
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

    def read(self):
        return self.grabbed, self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


def setup_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            track_id INTEGER,
            brand_name TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    return conn

def log_inspection(conn, track_id, class_name, conf):
    check_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO inspections (id, timestamp, track_id, brand_name, confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (check_id, timestamp, track_id, class_name, round(conf, 2)))
    conn.commit()
    return check_id

def run_system(source_input, headless=False):
    base_dir = Path(__file__).resolve().parent.parent

    # ── Model paths (OpenVINO first priority here) ──
    openvino_model_path = base_dir / "runs/detect/brand_experiment32/weights/best_openvino_model"
    onnx_model_path = base_dir / "runs/detect/brand_experiment32/weights/best.onnx"
    ncnn_model_path = base_dir / "runs/detect/brand_experiment32/weights/best_ncnn_model"
    pt_model_path = base_dir / "runs/detect/brand_experiment32/weights/best.pt"

    # ── Load Model (OpenVINO preferred) ──
    if openvino_model_path.exists():
        print(f"🚀 Loading OpenVINO model (320x320): {openvino_model_path}")
        model = YOLO(str(openvino_model_path), task='detect')
    elif onnx_model_path.exists():
        print(f"🔄 Falling back to ONNX model: {onnx_model_path}")
        model = YOLO(str(onnx_model_path), task='detect')
    elif ncnn_model_path.exists():
        print(f"🔄 Falling back to NCNN model: {ncnn_model_path}")
        model = YOLO(str(ncnn_model_path), task='detect')
    elif pt_model_path.exists():
        print(f"🔄 Falling back to PyTorch model: {pt_model_path}")
        model = YOLO(str(pt_model_path))
    else:
        print("⚠️ No custom model found! Using standard YOLOv11n.")
        model = YOLO("yolo11n.pt")

    db_path = base_dir / "data/inspections.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = setup_db(db_path)

    # ── Video capture ──
    if source_input.isnumeric():
        source_input = int(source_input)
        cap = VideoCaptureAsync(source_input)
        print("📹 Using threaded video capture for max FPS")
    else:
        cap = cv2.VideoCapture(source_input)

    logged_objects = set()
    frame_count = 0

    # FPS tracking
    fps_start = time.time()
    fps_counter = 0
    display_fps = 0.0

    print("🎥 Camera Active. Press 'Q' to quit.")
    print("⚡ Runtime: OpenVINO on ARM CPU")

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            break

        frame_count += 1
        # Skip every other frame for FPS boost on Pi
        if frame_count % 2 != 0:
            continue

        # Resize frame to reduce processing load
        small_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

        # imgsz=320 matches the exported OpenVINO model
        results = model.predict(small_frame, conf=0.85, imgsz=320, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = [-1] * len(boxes)

            clss = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                brand_name = model.names[int(cls)]
                x1, y1, x2, y2 = map(int, box)

                # Ignore tiny ghost bounding boxes
                if (x2 - x1) < 30 or (y2 - y1) < 30:
                    continue

                # Only show detections with ≥90% confidence
                if conf < 0.90:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"ID:{track_id} {brand_name}" if track_id != -1 else brand_name
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Log to DB only when model is very confident (≥95%)
                if conf >= 0.95 and track_id not in logged_objects:
                    uuid_code = log_inspection(conn, track_id, brand_name, conf)
                    print(f"📝 Logged to DB: {uuid_code} | {brand_name}")
                    logged_objects.add(track_id)

        # FPS counter
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(frame, f"OpenVINO FPS: {display_fps:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if not headless:
            cv2.imshow("ML Brand Detector [OpenVINO]", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if frame_count % 20 == 0:
                print(f"⚡ OpenVINO FPS: {display_fps:.1f}")

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    conn.close()
    print(f"\n✅ Session complete. Final FPS: {display_fps:.1f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Camera ID (0) or video path")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (for SSH/Pi)")
    args = parser.parse_args()
    run_system(args.source, headless=args.headless)
