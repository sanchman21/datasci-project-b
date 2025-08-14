import json
import cv2
import numpy as np
import os
import pandas as pd

def save_mask(mask, local_path, mask_name):
    os.makedirs(local_path, exist_ok=True)
    mask_path = os.path.join(local_path, mask_name)
    mask_shape = mask.shape
    np.save(mask_path, {'mask': np.packbits(mask), 'shape': mask_shape})
    return mask_path

def create_mask_from_polygons(height, width, polygons):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.array([(int(p['x']), int(p['y'])) for p in polygon], dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask

def process_ndjson(ndjson_path, images_dir):
    with open(ndjson_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            external_id = data['data_row']['external_id']
            height = data['media_attributes']['height']
            width = data['media_attributes']['width']
            
            project = data['projects']['clbzpv1vd19780803fz5kcark']
            labels = project['labels']
            if not labels:
                print(f"No labels found for {external_id}")
                continue
                
            label = labels[0]
            objects = label['annotations']['objects']
            
            nucleus_polygons = [obj['polygon'] for obj in objects if obj['name'] == 'mc_nuclear']
            cytoplasm_polygons = [obj['polygon'] for obj in objects if obj['name'] == 'mc_cytoplasm']
            
            nucleus_mask = create_mask_from_polygons(height, width, nucleus_polygons)
            cytoplasm_mask = create_mask_from_polygons(height, width, cytoplasm_polygons)
            
            mask = np.stack([nucleus_mask, cytoplasm_mask], axis=-1)
            
            mask_name = os.path.splitext(external_id)[0] + '.npy'
            mask_path = save_mask(mask, images_dir, mask_name)
            print(f"Saved mask for {external_id} to {mask_path}")

def generate_csv(images_dir, data_dir, csv_path):
    images_subdir = os.path.relpath(images_dir, data_dir)
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    data = []
    for image_file in image_files:
        image_path = os.path.join(images_subdir, image_file)
        mask_name = os.path.splitext(image_file)[0] + '.npy'
        mask_path = os.path.join(images_subdir, mask_name)
        data.append({'image_path': image_path, 'mask_path': mask_path})
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved to {csv_path}")

if __name__ == "__main__":
    ndjson_path = '../../data/labelbox2.ndjson'
    images_dir = '../../data/labelbox2/images'
    data_dir = '../../data'
    csv_path = '../datasets/CMML_Segmentation_Annotation_labelbox2.csv'
    
    process_ndjson(ndjson_path, images_dir)
    generate_csv(images_dir, data_dir, csv_path)