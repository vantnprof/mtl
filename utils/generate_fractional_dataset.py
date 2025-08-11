import os
import random
import shutil

# Set paths
base_path = "/work/datasets/DOTAv1"
img_src = os.path.join(base_path, "images/train")
lbl_src = os.path.join(base_path, "labels/train")

img_dst = os.path.join(base_path, "images/train12")
lbl_dst = os.path.join(base_path, "labels/train12")

# Create destination folders
os.makedirs(img_dst, exist_ok=True)
os.makedirs(lbl_dst, exist_ok=True)

# List all image files
image_files = [f for f in os.listdir(img_src) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()
random.seed(42)
subset = random.sample(image_files, 1*len(image_files) //2)

# Copy selected files
for img_file in subset:
    # Copy image
    src_img_path = os.path.join(img_src, img_file)
    dst_img_path = os.path.join(img_dst, img_file)
    shutil.copy2(src_img_path, dst_img_path)

    # Copy corresponding label
    label_file = os.path.splitext(img_file)[0] + ".txt"
    src_lbl_path = os.path.join(lbl_src, label_file)
    dst_lbl_path = os.path.join(lbl_dst, label_file)
    if os.path.exists(src_lbl_path):
        shutil.copy2(src_lbl_path, dst_lbl_path)
    else:
        print(f"Warning: label not found for {img_file}")

print(f"Sampled {len(subset)} images and corresponding labels.")

# Print statistics
total_images = len(image_files)
total_labels = len([f for f in os.listdir(lbl_src) if f.endswith('.txt')])
copied_images = len(os.listdir(img_dst))
copied_labels = len(os.listdir(lbl_dst))

print("\n--- Dataset Statistics ---")
print(f"Total training images: {total_images}")
print(f"Total training labels: {total_labels}")
print(f"Copied images to train13: {copied_images}")
print(f"Copied labels to train13: {copied_labels}")
print(f"Fraction of dataset copied: {copied_images / total_images:.2%}")