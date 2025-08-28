import os
import shutil

# Path to the devkit
devkit_path = '/work/datasets/imagenet1k/ILSVRC2012_devkit_t12'

# Path to the validation images
val_img_path = '/work/datasets/imagenet1k/val'

# Read the ground truth file
with open(os.path.join(devkit_path, 'data', 'ILSVRC2012_validation_ground_truth.txt'), 'r') as f:
    val_ground_truth = [int(line.strip()) for line in f.readlines()]

# Read the meta file to map class ID to synset ID (folder name)
with open(os.path.join(devkit_path, 'data', 'meta.mat'), 'rb') as f:
    import scipy.io
    meta = scipy.io.loadmat(f)
    id_to_synset = {int(s[0][0][0]): s[0][1][0] for s in meta['synsets']}

# Move images to their respective class folders
for i, img_filename in enumerate(sorted(os.listdir(val_img_path))):
    class_id = val_ground_truth[i]
    synset_id = id_to_synset[class_id]

    # Create folder if it doesn't exist
    class_folder = os.path.join(val_img_path, synset_id)
    os.makedirs(class_folder, exist_ok=True)

    # Move the image
    shutil.move(os.path.join(val_img_path, img_filename), class_folder)

print("Validation set prepared.")