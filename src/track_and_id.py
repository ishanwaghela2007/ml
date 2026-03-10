import cv2
import argparse
import uuid
import csv
import datetime
import os
import sqlite3
from pathlib import Path
from ultralytics import YOLO

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

def run_system(source_input):
    base_dir = Path(__file__).resolve().parent.parent
    
    # Paths to your trained weights (prioritizing lightweight versions for Raspberry Pi)
    onnx_model_path = base_dir / "runs/detect/brand_experiment2/weights/best.onnx"
    ncnn_model_path = base_dir / "runs/detect/brand_experiment2/weights/best_ncnn_model"
    tflite_model_path = base_dir / "runs/detect/brand_experiment2/weights/best_saved_model/best_float32.tflite"
    pt_model_path = base_dir / "runs/detect/brand_experiment2/weights/best.pt"
    log_path = base_dir / "logs/inspection_history.csv"
    
    # Load Model
    if onnx_model_path.exists():
        print(f"🚀 Loading optimized ONNX model: {onnx_model_path}")
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

    # Webcam or Video
    if source_input.isnumeric():
        source_input = int(source_input)
    
    cap = cv2.VideoCapture(source_input)
    
    logged_objects = set()
    frame_count = 0

    print("🎥 Camera Active. Press 'Q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame_count += 1
        # Optional: uncomment the next 2 lines to skip every other frame for a huge FPS boost
        # if frame_count % 2 != 0:
        #     continue

        # Using a much higher confidence threshold (0.70) to strictly prevent ghost detections when the frame is empty
        # Explicitly setting imgsz=640 tells the YOLO engine to correctly calculate bounding box scaling
        results = model.predict(frame, conf=0.70, imgsz=640, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            
            # Extract tracking IDs safely (predict doesn't assign IDs, so default to -1)
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = [-1] * len(boxes)

            clss = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                brand_name = model.names[int(cls)]
                
                # Draw
                x1, y1, x2, y2 = map(int, box)
                
                # Further prevent ghost detections by ignoring tiny random bounding boxes
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width < 40 or box_height < 40:
                    continue
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label_text = f"ID:{track_id} {brand_name}" if track_id != -1 else f"{brand_name}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Log to DB (Also checking against a higher threshold here just to be safe)
                if conf > 0.80 and track_id not in logged_objects:
                    uuid_code = log_inspection(conn, track_id, brand_name, conf)
                    print(f"📝 Logged to DB: {uuid_code} | {brand_name}")
                    logged_objects.add(track_id)

        cv2.imshow("ML Brand Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Camera ID (0) or video path")
    args = parser.parse_args()
    run_system(args.source)
