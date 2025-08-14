import os
import pandas as pd
import argparse

BASE_DIR = "../../"
SEGMENTED_DIR = os.path.join(BASE_DIR, "data", "segmented_images_monocyte_new_normals")
INPUT_CSV_PATH = os.path.join("../", "datasets", "monocyte_new_normals_segmented.csv")
MONOCYTE_NEW_NORMALS_CSV = os.path.join("../", "datasets", "monocyte_new_normals_segmented.csv")
MONOCYTE_CSV = os.path.join("../", "datasets", "monocyte_reassigned_segmented.csv")

def clean_segmented_csv(input_csv_path, segmented_dir):
    df = pd.read_csv(input_csv_path)
    segmented_paths = [os.path.join(segmented_dir, os.path.basename(path)) for path in df['image_path']]
    existing_paths = [path for path in segmented_paths if os.path.exists(path)]
    missing_paths = [path for path in segmented_paths if not os.path.exists(path)]
    keep_indices = [i for i, path in enumerate(segmented_paths) if path in existing_paths]
    original_count = len(df)
    df_cleaned = df.iloc[keep_indices].reset_index(drop=True)
    deleted_count = original_count - len(df_cleaned)
    df_cleaned.to_csv(input_csv_path, index=False)
    print(f"Saved cleaned dataset to {input_csv_path}")
    print(f"Deleted {deleted_count} rows due to missing segmented images.")
    if missing_paths:
        print(f"Missing segmented images: {len(missing_paths)}")
        for path in missing_paths[:5]:
            print(f"  - {path}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean segmented CSV for monocyte datasets.")
    parser.add_argument('--data_type', type=str, choices=['monocyte', 'monocyte_new_normals'], default='monocyte_new_normals', help='Type of dataset to clean')
    args = parser.parse_args()
    if args.data_type == 'monocyte_new_normals':
        input_csv_path = MONOCYTE_NEW_NORMALS_CSV
    elif args.data_type == 'monocyte':
        input_csv_path = MONOCYTE_CSV
    else:
        raise ValueError(f"Unknown data_type: {args.data_type}")
    segmented_dir = os.path.join(BASE_DIR, "data", f"segmented_images_{args.data_type}")
    clean_segmented_csv(input_csv_path, segmented_dir)