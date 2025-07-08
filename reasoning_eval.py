import json
import os
import io
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from utils.utils import get_prediction_result

# Load dataset JSON file
def load_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["rows"]

# Compute IoU
def compute_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return intersection / union if union > 0 else 0

# Process the entire dataset
def process_dataset(df, dataset_root, molmo_output_root, output_json_path):
    results = []
    correct_count = 0
    
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
                
    processed_scenes = {tuple(r[:3]) for r in results}

    for _, row in df.iterrows():
        # scene_id, obj_id, annotation, gt_id, _ = row
        scene_id = row["sceneId"]
        obj_id = row["queryObjId"]
        annotation = row["annotation"]
        gt_id = row["groundTruthObjIds"]
        image_bytes = row["image"]["bytes"]
        
        # Dataset object IDs start from 0, while pointed object IDs start from 1. Need to increment by 1.
        obj_id = int(obj_id) + 1
        
        if (scene_id, obj_id, annotation) in processed_scenes:
            print(f"Skipping Scene {scene_id}, Object {obj_id}, already processed.")
            continue
        
        print(f"Processing Scene {scene_id}, Object {obj_id}...")

        image_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        dataset_root = Path(dataset_root)
        gt_mask_path = dataset_root / f"{scene_id}.npz"
        molmo_folder = Path(molmo_output_root) / f"scene{scene_id}"
        molmo_image_path = molmo_folder/f"{scene_id}.png"

        if not gt_mask_path.exists():
            print(f"⚠️ GT mask file {gt_mask_path} not found!")
            continue

        gt_data = np.load(gt_mask_path, allow_pickle=True)
        gt_masks = gt_data["instances_objects.npy"]

        try:
            gt_ids = [int(x) + 1 for x in str(gt_id).split(",")]
        except ValueError:
            print(f"⚠️ Invalid GT ID format: {gt_id} → Skipping this scene.")
            continue
        
        valid_gt = False
        # image_path include original image for LangSAM, molmo_folder include labeled image (for GPT) and points coordinates (for LangSAM post-process).

        try:
            mask_pred = get_prediction_result(image, molmo_folder, scene_id, annotation)
            
            if isinstance(mask_pred, dict) and "error" in mask_pred:
                print(f"Scene {scene_id}, Object {obj_id} failed: {mask_pred['error']}")
                results.append([scene_id, obj_id, annotation, 0])
                continue
        
        except Exception as e:
            print(f"❌ Unexpected error in Scene {scene_id}, Object {obj_id}: {e}")
            results.append([scene_id, obj_id, annotation, 0])
            continue 

        for gt_id in gt_ids:
            if 0< gt_id <= len(gt_masks):
                gt_mask = (gt_masks == gt_id).astype(np.uint8)
                iou = compute_iou(mask_pred, gt_mask)
                if iou>=0.5:
                    valid_gt = True
                    break
            else:
                print(f"⚠️ GT ID {gt_id} is out of range for Scene {scene_id}.")     
                
        is_correct = 1 if valid_gt else 0
        correct_count += is_correct           
        results.append([scene_id, obj_id, annotation, is_correct])
        
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"✅ Scene {scene_id}, Object {obj_id}: IoU={iou:.4f}, {'✔' if is_correct else '❌'}")

    # Calculate accuracy
    accuracy = correct_count / len(df) if len(df) else 0
    print(f"\n🎯 Accuracy: {accuracy:.2%} ({correct_count}/{len(df)})")
    print(f"📂 Results saved to {output_json_path}")


if __name__ == "__main__":
    # Path of dataset
    PARQUET_FILES = [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet"
    ]
    NPZ_DIR = "data/npz_file"
    molmo_output_root = "data/output/molmo_output"

    # Load the dataset
    df = pd.concat([pd.read_parquet(p) for p in PARQUET_FILES])

    # Select split to run
    selected_split = '0'  # Change this value to 0, 1, or 2

    df = df[df['split'] == selected_split]  # Filter dataset based on selected split

    # Select subsets for testing
    easy_no_ambi = df[(df['difficulty'] == 'Easy') & (df['ambiguious'] == False)]
    easy_yes_ambi = df[(df['difficulty'] == 'Easy') & (df['ambiguious'] == True)]
    medium_no_ambi = df[(df['difficulty'] == 'Medium') & (df['ambiguious'] == False)]
    medium_yes_ambi = df[(df['difficulty'] == 'Medium') & (df['ambiguious'] == True)]
    hard_no_ambi = df[(df['difficulty'] == 'Hard') & (df['ambiguious'] == False)]
    hard_yes_ambi = df[(df['difficulty'] == 'Hard') & (df['ambiguious'] == True)]

    # Choose a subset for testing (e.g., easy_no_ambi)
    subset_name = "easy_no_ambi"  
    selected_subset = eval(subset_name)  
    print(f"Total number of examples in [SPLIT {selected_split}] of [SET {subset_name}]: {len(selected_subset)}")

    # Run test
    dataset_root = NPZ_DIR
    output_json_path = f"data/output/out_{subset_name}.json"
    process_dataset(selected_subset, dataset_root, molmo_output_root, output_json_path)