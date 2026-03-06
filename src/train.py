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

    print(f"🏋️ Starting training...")
    model.train(
        data=str(yaml_path),
        epochs=300,
        imgsz=640,
        device="cpu",
        project=str(base_dir / "runs" / "detect"),
        name="brand_experiment"
    )

if __name__ == '__main__':
    train_model()
