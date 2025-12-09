from ultralytics import YOLO
import matplotlib.pyplot as plt

# 1. Load trained model – change this to the run that was just saved
model = YOLO(r"runs/detect/train3/weights/best.pt")  # or use forward slashes: "runs/detect/train3/weights/best.pt"

# 2. Path to any image you want to test
img_path = r"data/debris-dataset/debris-detection/val/0.jpg"  # change index if 0.jpg doesn't exist

results = model(img_path, conf=0.4)[0]

annotated = results.plot()[:, :, ::-1]
plt.imshow(annotated)
plt.axis("off")
plt.title("Debris Detection Result")
plt.show()
