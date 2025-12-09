# Space Debris Detection Using YOLOv8

This project detects space debris from satellite imagery using a custom-trained YOLOv8 model.

## 🚀 Features
- Converts dataset annotations to YOLO format  
- Trains YOLOv8 on a debris detection dataset  
- Tests model on new images  
- Fully reproducible pipeline (VS Code)

## 📁 Project Structure
space_debris_yolo/
├── data/ # dataset (ignored in GitHub)
├── yolo_dataset/ # YOLO formatted labels (ignored)
├── runs/ # training outputs (ignored)
├── convert_to_yolo.py
├── train_yolo_debris.py
├── test_yolo_debris.py
├── debris.yaml
└── README.md


## 🧠 Model Training  Testing
:-python train_yolo_debris.py   
## testing
python test_yolo_debris.py


