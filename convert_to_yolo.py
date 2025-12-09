import os
import ast
import pandas as pd
from PIL import Image

# Paths
BASE = r"data/debris-dataset/debris-detection"
TRAIN_CSV = os.path.join(BASE, "train.csv")
VAL_CSV = os.path.join(BASE, "val.csv")

TRAIN_IMG_DIR = os.path.join(BASE, "train")
VAL_IMG_DIR = os.path.join(BASE, "val")

OUT = "yolo_dataset"
os.makedirs(f"{OUT}/images/train", exist_ok=True)
os.makedirs(f"{OUT}/labels/train", exist_ok=True)
os.makedirs(f"{OUT}/images/val", exist_ok=True)
os.makedirs(f"{OUT}/labels/val", exist_ok=True)

def convert_split(csv_path, img_dir, out_split):
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        img_name = f"{row['ImageID']}.jpg"
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print("Missing image:", img_path)
            continue

        # copy image
        os.system(f'copy "{img_path}" "{OUT}/images/{out_split}\\"')

        # load image to get width/height
        img = Image.open(img_path)
        w, h = img.size

        # parse bbox list
        bboxes = ast.literal_eval(row["bboxes"])

        label_path = f"{OUT}/labels/{out_split}/{row['ImageID']}.txt"
        with open(label_path, "w") as f:
            for box in bboxes:
                xmin, ymin, xmax, ymax = box

                # convert to YOLO format
                x_center = (xmin + xmax) / 2 / w
                y_center = (ymin + ymax) / 2 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h

                class_id = 0  # only one class: debris
                f.write(f"{class_id} {x_center} {y_center} {bw} {bh}\n")

    print(f"Finished converting {out_split} split.")

# Convert both splits
convert_split(TRAIN_CSV, TRAIN_IMG_DIR, "train")
convert_split(VAL_CSV, VAL_IMG_DIR, "val")
