import os
import cv2
import pandas as pd
from pathlib import Path

# Target YOLO dataset structure

# dataset/
# ├── images/
# │   ├── train/
# │   │   ├── img1.jpg
# │   │   ├── img2.jpg
# │   ├── val/
# │       ├── img3.jpg
# │       ├── img4.jpg
# ├── labels/
# │   ├── train/
# │   │   ├── img1.txt
# │   │   ├── img2.txt
# │   ├── val/
# │       ├── img3.txt
# │       ├── img4.txt


# Below is just specific for the GTex liver training data, given that annotations are already spilt into train/val/test at the first place
# !!path name all subject to modification, please check your folder structure for specific!!

#TODO: refactor so not hard coded and linked in nuc train yolo or in read me to set up daaset structure like this? 
# or have the make tile dataset do this from the start?
 

base_output = "projects/multiorgan/dataset/nuclei/ovary"
project_path = Path('projects/multiorgan')

splits = {
    "train": project_path / "annotations/nuclei/ovary/train_nuclei.csv",
    "val":   project_path / "annotations/nuclei/ovary/val_nuclei.csv",
    "test":  project_path / "annotations/nuclei/ovary/test_nuclei.csv",
}

# Build class mapping once (important!)
all_dfs = [pd.read_csv(csv, header=None) for csv in splits.values()]
full_df = pd.concat(all_dfs)
full_df.columns = ["image_path", "x1", "y1", "x2", "y2", "class_name"] # standard retinanet COCO data annotation structure
# Drop NaN class names (empty tiles) before sorting -- build empty txt files later 
classes = sorted(full_df["class_name"].dropna().unique())
class_to_id = {name: i for i, name in enumerate(classes)} # YOLO used class id instead of class name 

for split, csv_path in splits.items():
    df = pd.read_csv(csv_path, names=["image_path", "x1", "y1", "x2", "y2", "class_name"], header=None)
    grouped = df.groupby("image_path") # grouped by single image 

    for img_path, group in grouped:
        img_path = os.path.join(project_path,img_path)
        img = cv2.imread(img_path) #read in img to extract raw image height and weight
        if img is None:
            print(f"Failed to read {img_path}")
            continue

        h, w = img.shape[:2]
        img_name = os.path.basename(img_path)

        # Moving image to the destination 
        
        # Destination paths for images
        dst_img = os.path.join(base_output, "images", split, img_name)
        # Destination paths for annotation label txt
        dst_label = os.path.join(
            base_output, "labels", split, img_name.rsplit(".", 1)[0] + ".txt"
        )

        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)

        # Copy image
        cv2.imwrite(dst_img, img)  # or can be changed to moving file instead of copying

        # Write label
        with open(dst_label, "w") as f:
            for _, row in group.iterrows():
                # Skip empty-tile rows (NaN class) — empty .txt file means background
                # (but as still opened and written empty txt file, just pass over the rest of the code)
                if pd.isna(row["class_name"]):
                    continue

                x1, y1, x2, y2 = row[["x1", "y1", "x2", "y2"]]
                cls_id = class_to_id[row["class_name"]]

                # Convert - normalise to [0,1]
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(f"{cls_id} {x_center} {y_center} {bw} {bh}\n")


