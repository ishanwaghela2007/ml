from ultralytics import YOLO
import cv2
import time
from pathlib import Path

# --- PI NCNN INFERENCE (WITH DISPLAY) ---
# This uses Ultralytics to run the NCNN model for easy box drawing.

if __name__ == "__main__":
    # Path to your exported NCNN folder
    MODEL_PATH = "best_ncnn_model" 
    
    # --- DISPLAY IS NOW ON ---
    HEADLESS = False 
    
    print(f"🚀 Initializing Unified Tube Detection (NCNN Mode)...")
    try:
        # Load the NCNN model via Ultralytics (handles boxes automatically)
        model = YOLO(MODEL_PATH, task='detect')
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            start = time.time()
            
            # Run NCNN inference (640x640 is standard)
            results = model.predict(frame, conf=0.5, imgsz=640, verbose=False)
            
            fps = 1 / (time.time() - start)
            
            if not HEADLESS:
                # Plot results (draws the lines and labels automatically)
                annotated_frame = results[0].plot()
                
                # Show FPS on screen
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Pi Edge Live View", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print(f"\rDetecting... FPS: {fps:.2f}", end="")
                
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Hint: Ensure 'ultralytics' is installed and your model folder is correct.")
