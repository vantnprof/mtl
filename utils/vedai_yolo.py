import os
import shutil
import yaml
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Define paths
root_dir = '/work/datasets/vedai/raw/512'
output_dir = '/work/datasets/vedai/processed/512'
annotations_dir = os.path.join(root_dir, 'Annotations512')
images_dir = os.path.join(root_dir, 'Vehicules512')

# Define class names and mapping
class_names = ['car', 'pickup', 'camping_car', 'truck', 'other', 'tractor', 'boat', 'van', 'bus', 'bicycle', 'motorcycle']
class_id_map = {
    1: 0, 2: 1, 4: 2, 5: 3, 7: 4,
    8: 5, 9: 6, 10: 7, 11: 8, 23: 9, 31: 10
}

def parse_annotation(txt_path, img_w, img_h):
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            raw_cls_id = int(parts[3])
            if raw_cls_id not in class_id_map:
                continue  # skip unused classes
            cls_id = class_id_map[raw_cls_id]
            x_coords = list(map(float, parts[6:10]))
            y_coords = list(map(float, parts[10:]))
            x1, x2, x3, x4 = [x / img_w for x in x_coords]
            y1, y2, y3, y4 = [y / img_h for y in y_coords]
            boxes.append([cls_id, x1, y1, x2, y2, x3, y3, x4, y4])
    return boxes

# Image dimensions
img_w, img_h = 512, 512

# Process each fold
for fold in range(1, 11):
    train_images_out_dir = os.path.join(output_dir, f'fold{fold:02d}', 'images/train')
    train_labels_out_dir = os.path.join(output_dir, f'fold{fold:02d}', 'labels/train')
    os.makedirs(train_images_out_dir, exist_ok=True)
    os.makedirs(train_labels_out_dir, exist_ok=True)

    # Read fold file
    fold_file = os.path.join(annotations_dir, f'fold{fold:02d}.txt')
    if not os.path.exists(fold_file):
        print(f"Fold file {fold_file} does not exist. Skipping fold {fold}.")
        continue

    with open(fold_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]

    for img_id in image_ids:
        img_path = os.path.join(images_dir, img_id + "_co.png")
        ann_path = os.path.join(annotations_dir, img_id + ".txt")

        out_img_path = os.path.join(train_images_out_dir, img_id + ".jpg")
        out_lbl_path = os.path.join(train_labels_out_dir, img_id + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        # Convert annotation
        if os.path.exists(ann_path):
            boxes = parse_annotation(ann_path, w, h)
            with open(out_lbl_path, "w") as f:
                for box in boxes:
                    f.write("{} {}\n".format(box[0], " ".join(f"{x:.6f}" for x in box[1:])))
        img.save(out_img_path, "JPEG")
    print(f"Processed fold {fold} train with: {len(image_ids)} images")


    val_images_out_dir = os.path.join(output_dir,  f'fold{fold:02d}', 'images/val')
    val_labels_out_dir = os.path.join(output_dir,  f'fold{fold:02d}', 'labels/val')
    os.makedirs(val_images_out_dir, exist_ok=True)
    os.makedirs(val_labels_out_dir, exist_ok=True)

    test_fold_file = os.path.join(annotations_dir, f'fold{fold:02d}test.txt')
    if not os.path.exists(test_fold_file):
        print(f"Fold file {test_fold_file} does not exist. Skipping fold {fold}.")
        continue

    with open(test_fold_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]

    for img_id in image_ids:
        img_path = os.path.join(images_dir, img_id + "_co.png")
        ann_path = os.path.join(annotations_dir, img_id + ".txt")

        out_img_path = os.path.join(val_images_out_dir, img_id + ".jpg")
        out_lbl_path = os.path.join(val_labels_out_dir, img_id + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        # Convert annotation
        if os.path.exists(ann_path):
            boxes = parse_annotation(ann_path, w, h)

            with open(out_lbl_path, "w") as f:
                for box in boxes:
                    f.write("{} {}\n".format(box[0], " ".join(f"{x:.6f}" for x in box[1:])))
        img.save(out_img_path, "JPEG")
    print(f"Processed fold {fold} test with: {len(image_ids)} images")

    # Create YAML file
    yaml_path = os.path.join('./src/yolo11/data/', f"vedai_fold{fold:02}.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.join(output_dir, f'fold{fold:02d}')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test:\n")
        f.write("\n# Classes\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")
    print(f"Dataset config written to: {yaml_path}")


