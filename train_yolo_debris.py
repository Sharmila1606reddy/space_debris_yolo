from ultralytics import YOLO

# Use the smallest YOLO model (fastest)
model = YOLO("yolov8n.pt")  # 'n' = nano

model.train(
    data="debris.yaml",  # <-- make sure the file name matches exactly
    epochs=5,            # fewer epochs = faster (you can later increase to 10+)
    imgsz=416,           # smaller image size = faster than 640
    batch=4,             # reduce batch size to avoid RAM issues on CPU
    device="cpu",        # FORCE CPU (no GPU / CUDA)
    workers=0,           # avoid Windows dataloader overhead
    cache=True,          # cache images in RAM after first epoch (if enough RAM)
    fraction=0.2,        # use only 20% of data to speed up (remove later for full training)
)
