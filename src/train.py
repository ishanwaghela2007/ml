from ultralytics import YOLO
import os
from pathlib import Path

def train_model():
    print("🚀 Loading YOLOv11 Nano model...")
    model = YOLO("yolo11n.pt") 

    # Robust path handling
    base_dir = Path(__file__).resolve().parent.parent # This points to 'ml/main'
    data_dir = base_dir / 'data' / 'dataset(tubes)'

    # Read classes from classes.txt
    classes_file = data_dir / 'classes.txt'
    if not classes_file.exists():
        print(f"Error: classes.txt not found at {classes_file}")
        return

    with open(classes_file, 'r') as f:
        company_classes = {i: line.strip() for i, line in enumerate(f.readlines()) if line.strip()}

    print("Detected classes:", company_classes)

    # Create YAML config
    names_yaml = "\n".join([f"  {k}: '{v}'" for k, v in company_classes.items()])
    
    yaml_content = f'''
path: {data_dir.as_posix()} 
train: images
val: images

names:
{names_yaml}
'''
    
    yaml_path = data_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"🏋️ Starting training with heavy augmentation (small dataset = need more augmentation)...")
    model.train(
        data=str(yaml_path),
        epochs=500,              # More epochs: small dataset needs more passes to learn
        imgsz=640,
        batch=4,                 # Small batch for small dataset (prevents overfitting noise)
        device="cpu",
        patience=100,            # Early stopping: stop if no improvement for 100 epochs
        
        # ── Heavy Augmentation (critical for only 17 images!) ──
        augment=True,
        hsv_h=0.02,              # Hue shift — helps learn color variations
        hsv_s=0.7,               # Saturation shift
        hsv_v=0.5,               # Brightness shift  
        degrees=15.0,            # Rotation (tube can be held at angles)
        translate=0.2,           # Shift left/right/up/down
        scale=0.6,               # Zoom in/out
        shear=5.0,               # Slight shear distortion
        flipud=0.1,              # Vertical flip (small chance)
        fliplr=0.5,              # Horizontal flip
        mosaic=1.0,              # Mosaic augmentation (combines 4 images into 1)
        mixup=0.2,               # MixUp augmentation (blends 2 images)
        copy_paste=0.1,          # Copy-paste augmentation
        erasing=0.2,             # Random erasing (forces model to not rely on one part)
        
        # ── Regularization (prevent overfitting on 17 images) ──
        dropout=0.1,             # Dropout for generalization
        weight_decay=0.001,      # L2 regularization
        
        # ── Learning rate ──
        lr0=0.01,                # Initial learning rate
        lrf=0.001,               # Final learning rate (cosine decay)
        warmup_epochs=10,        # Gradual warmup
        
        # ── Output ──
        project=str(base_dir / "runs" / "detect"),
        name="brand_experiment3"
    )

    print("✅ Training complete!")
    print(f"📁 Weights saved to: {base_dir / 'runs/detect/brand_experiment3/weights/'}")

if __name__ == '__main__':
    train_model()
