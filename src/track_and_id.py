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
# Threaded Video Capture — keeps frame grabbing off the inference
# thread so the Pi never stalls waiting for the camera.
# ───────────────────────────────────────────────────────────────────
class VideoCaptureAsync:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        # Force 480p to reduce the amount of pixels the Pi has to handle
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab the latest frame
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
    
    # Paths to your trained weights (prioritizing lightweight versions for Raspberry Pi)
    onnx_model_path = base_dir / "runs/detect/brand_experiment32/weights/best.onnx"
    ncnn_model_path = base_dir / "runs/detect/brand_experiment32/weights/best_ncnn_model"
    tflite_model_path = base_dir / "runs/detect/brand_experiment32/weights/best_saved_model/best_float32.tflite"
    pt_model_path = base_dir / "runs/detect/brand_experiment32/weights/best.pt"
    
    # Load Model
    if onnx_model_path.exists():
        print(f"🚀 Loading optimized ONNX model (320x320): {onnx_model_path}")
        model = YOLO(str(onnx_model_path), task='detect')
    elif ncnn_model_path.exists():
        print(f"🚀 Loading highly-optimized NCNN model: {ncnn_model_path}")
        model = YOLO(str(ncnn_model_path), task='detect')
    elif tflite_model_path.exists():
        print(f"🚀 Loading lightweight TFLite model: {tflite_model_path}")
        model = YOLO(str(tflite_model_path), task='detect')
    elif pt_model_path.exists():
        print(f"✅ Loading standard PyTorch model: {pt_model_path}")
        model = YOLO(str(pt_model_path))
    else:
        print("⚠️ Custom model not found! Using standard YOLOv11n.")
        model = YOLO("yolo11n.pt")

    db_path = base_dir / "data/inspections.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = setup_db(db_path)

    # Webcam or Video — use threaded capture for live camera on the Pi
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
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            break

        frame_count += 1
        # ── Skip every other frame for a large FPS boost on the Pi ──
        if frame_count % 2 != 0:
            continue

        # ── Resize frame before inference to reduce processing load ──
        # The model was exported at 320x320, so we match that scale
        small_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

        # Inference with tracking
        results = model.track(small_frame, conf=0.90, imgsz=320, persist=True, verbose=False)

        detections = []
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
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width < 30 or box_height < 30:
                    continue
                
                # Filter confidence
                if conf < 0.95:
                    continue

                detections.append((brand_name, conf, (x1, y1, x2, y2), track_id))

                if not headless:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"ID:{track_id} {brand_name}" if track_id != -1 else brand_name
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Persistence & Gap Logic
        current_time = time.time()
        if not hasattr(run_system, 'logged_ids'):
            run_system.logged_ids = set()
        if not hasattr(run_system, 'last_action_time'):
            run_system.last_action_time = 0

        global_gap = 5.0 # 5 seconds gap
        time_since_last_log = current_time - run_system.last_action_time

        if detections and time_since_last_log > global_gap:
            for brand_name, conf, _, track_id in detections:
                if track_id != -1:
                    if track_id in run_system.logged_ids:
                        continue
                    
                    run_system.logged_ids.add(track_id)
                    uuid_code = log_inspection(conn, track_id, brand_name, conf)
                    run_system.last_action_time = current_time
                    print(f"📝 [LOGGED] {brand_name} (ID:{track_id}). UUID: {uuid_code}. Gap active for {global_gap}s")
                    break
                # Skip logging if track_id is -1

        # ── FPS counter ──
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()
        
        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if not headless:
            cv2.imshow("ML Brand Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # In headless mode, just print FPS periodically
            if frame_count % 20 == 0:
                print(f"⚡ FPS: {display_fps:.1f}")

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
