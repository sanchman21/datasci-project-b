import pandas as pd
import os
import shutil
import random
from pathlib import Path

def main():
    csv_path = "../datasets/cmml_500_segmented.csv"
    df = pd.read_csv(csv_path)
    correct_indices = []
    
    with open("./experiments/segmentation/results/correct_indices.txt", "r") as f:
        for line in f:
            indices = [int(x.strip()) for x in line.split(",")]
            correct_indices.extend(indices)
            
    print(f"Total rows in CSV: {len(df)}")
    print(f"Correct indices count: {len(correct_indices)}")
    all_indices = list(range(len(df)))
    incorrect_indices = [idx for idx in all_indices if idx not in correct_indices]
    print(f"Incorrect indices count: {len(incorrect_indices)}")
    num_to_select = min(200, len(incorrect_indices))
    selected_incorrect_indices = random.sample(incorrect_indices, num_to_select)
    print(f"Selected {num_to_select} incorrect indices")
    target_dir = "../../data/labelbox2/images"
    os.makedirs(target_dir, exist_ok=True)
    copied_count = 0
    
    for idx in selected_incorrect_indices:
        try:
            original_path = df.iloc[idx]['original_path']
            if os.path.exists(original_path):
                _, ext = os.path.splitext(original_path)
                new_filename = f"cmml_{idx}{ext}"
                target_path = os.path.join(target_dir, new_filename)
                shutil.copy2(original_path, target_path)
                copied_count += 1
                print(f"Copied {original_path} -> {target_path}")
            else:
                print(f"File not found: {original_path}")
        except Exception as e:
            print(f"Error copying index {idx}: {e}")
            
    print(f"Successfully copied {copied_count} images")
    labelbox_source_dir = "../../data/labelbox/data/CMML Segmentation Annotation"
    labelbox_copied_count = 0
    
    if os.path.exists(labelbox_source_dir):
        for filename in os.listdir(labelbox_source_dir):
            file_path = os.path.join(labelbox_source_dir, filename)
            if not os.path.isfile(file_path):
                continue
            if filename.endswith('_label_pv.jpg') or filename.endswith('_viz.jpg'):
                continue
            if filename.endswith('.npy') or filename.endswith('.jpg'):
                target_path = os.path.join(target_dir, filename)
                try:
                    shutil.copy2(file_path, target_path)
                    labelbox_copied_count += 1
                    print(f"Copied labelbox file: {filename}")
                except Exception as e:
                    print(f"Error copying labelbox file {filename}: {e}")
    else:
        print(f"Labelbox source directory not found: {labelbox_source_dir}")
        
    print(f"Successfully copied {labelbox_copied_count} labelbox files")
    print(f"Total files in target directory: {copied_count + labelbox_copied_count}")
    
if __name__ == "__main__":
    main()
