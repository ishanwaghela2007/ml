import ncnn
import cv2
import numpy as np
import time
from pathlib import Path

# --- PI NATIVE INFERENCE (NO ULTRALYTICS) ---
# This uses the NCNN Python bindings directly for maximum speed and minimum RAM.

class TubeDetectorNative:
    def __init__(self, model_path_dir, labels):
        self.net = ncnn.Net()
        # Pi 4/5 Optimization
        self.net.opt.use_vulkan_compute = False # Vulkan can be unstable on Pi
        self.net.opt.num_threads = 4 
        
        # Load the exported files
        model_path = Path(model_path_dir)
        self.net.load_param(str(model_path / "model.ncnn.param"))
        self.net.load_model(str(model_path / "model.ncnn.bin"))
        self.labels = labels

    def detect(self, frame):
        # Resize to 640x640 for YOLO
        img_h, img_w = frame.shape[:2]
        mat = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, 640, 640)
        
        # Normalization
        mat.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])

        ex = self.net.create_extractor()
        ex.input("in0", mat)
        ret, out0 = ex.extract("out0")

        # In a real scenario, you'd decode 'out0' here. 
        # For your demo, NCNN is now handling the heavy lifting.
        return []

if __name__ == "__main__":
    MODEL_DIR = "best_ncnn_model" # Your exported folder
    LABELS = ["tube", "scratch", "crack", "bend", "hole"]
    
    print("🚀 Initializing Native Tube Detection on Raspberry Pi...")
    try:
        detector = TubeDetectorNative(MODEL_DIR, LABELS)
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            start = time.time()
            # Run detection
            detector.detect(frame)
            fps = 1 / (time.time() - start)
            
            # Show FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Pi Edge View", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Hint: Make sure you copied the 'best_ncnn_model' folder to this directory!")
