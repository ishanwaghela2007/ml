import cv2
import argparse
import uuid
import csv
import datetime
import os
from pathlib import Path
from ultralytics import YOLO

def setup_logger(log_file):
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Check_UUID', 'Timestamp', 'Object_Track_ID', 'Brand_Name', 'Confidence'])

def log_inspection(log_file, track_id, class_name, conf):
    check_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([check_id, timestamp, track_id, class_name, f"{conf:.2f}"])
    return check_id

def run_system(source_input):
    base_dir = Path(__file__).resolve().parent.parent
    
    # Path to your trained weights
    model_path = base_dir / "runs/detect/brand_experiment2/weights/best.pt"
    log_path = base_dir / "logs/inspection_history.csv"
    
    # Load Model
    if model_path.exists():
        print(f"✅ Loading custom trained model: {model_path}")
        model = YOLO(model_path)
    else:
        print("⚠️ Custom model not found! Using standard YOLOv11n.")
        model = YOLO("yolo11n.pt")

    setup_logger(log_path)

    # Webcam or Video
    if source_input.isnumeric():
        source_input = int(source_input)
    
    cap = cv2.VideoCapture(source_input)
    logged_objects = set()

    print("🎥 Camera Active. Press 'Q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Using standard prediction with a realistic confidence threshold to prevent ghost detections
        results = model.predict(frame, conf=0.60, verbose=False)

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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label_text = f"ID:{track_id} {brand_name}" if track_id != -1 else f"{brand_name}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Log to CSV
                if conf > 0.6 and track_id not in logged_objects:
                    uuid_code = log_inspection(log_path, track_id, brand_name, conf)
                    print(f"📝 Logged: {uuid_code} | {brand_name}")
                    logged_objects.add(track_id)

        cv2.imshow("ML Brand Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Camera ID (0) or video path")
    args = parser.parse_args()
    run_system(args.source)
