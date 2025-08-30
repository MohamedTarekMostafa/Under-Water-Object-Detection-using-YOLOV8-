#  Aquarium Object Detection with YOLOv8

This project applies **YOLOv8 (You Only Look Once v8)** for object detection on the **Aquarium Dataset**.  
The dataset includes multiple underwater species such as fish, jellyfish, penguins, puffins, sharks, starfish, and stingrays.  

## 🚀 Project Overview
- **Goal**: Train and evaluate a custom YOLOv8 model to detect different classes of marine life.  
- **Dataset**: Aquarium Dataset (Kaggle).  
- **Framework**: Ultralytics YOLOv8.  

##  Dataset Structure
aquarium_pretrain/
│── train/
│ └── images/, labels/
│── valid/
│ └── images/, labels/
│── test/
│ └── images/, labels/
│── data.yaml

## ⚙️ Model Training
We used **YOLOv8n (nano version)** for faster training.

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='/kaggle/working/data.yaml',
    epochs=100,
    imgsz=640,
    batch=80,
    workers=8,
    save_period=10,
    lr0=5e-4,
    optimizer='AdamW',
    augment=True,
    name='yolov8n_custom',
    weight_decay=0.01,
)
#Results
| Metric     | Value |
| ---------- | ----- |
| Precision  | 0.76  |
| Recall     | 0.70  |
| mAP\@50    | 0.77  |
| mAP\@50-95 | 0.52  |
#Inference
results = model.predict(source='test/images', conf=0.25)
#Test on a video
results = model.predict(source='test/video.mp4', conf=0.25, save=True)
#🔮 Future Work
Apply to oil & gas monitoring (e.g., leak detection, equipment inspection).
Use larger models (YOLOv8s/m/l) for higher accuracy.
👤 Author
Mohamed Tarek
Computer Engineer & AI Engineer


