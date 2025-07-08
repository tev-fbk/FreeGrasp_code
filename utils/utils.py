import os
import re
import torch
import base64
import numpy as np
import open3d as o3d
from PIL import Image
from matplotlib import pyplot as plt
from models.FGC_graspnet.utils.data_utils import create_point_cloud_from_depth_image

from utils.config import client, langsam_actor, api_key

def get_coordinates(labeled_text, goal_id):
    """
    Extract the (X, Y) coordinates for a given goal_id.
    """
    if isinstance(labeled_text, str):
        lines = labeled_text.strip().split("\n")[1:]  # Skip the header
    elif isinstance(labeled_text, list):
        lines = labeled_text[1:]
    
    for line in lines:
        parts = line.split()
        if int(parts[0]) == goal_id:
            return int(parts[1]), int(parts[2])
    return None  # Return None if not found


def get_goal_mask_with_index(masks, goal_coor):
    """
    Select the best mask (and its index) that contains the goal coordinate.
    If none contain the coordinate, select the closest one.
    """
    x, y = goal_coor
    best_mask = None
    selected_index = None
    min_distance = float('inf')

    for idx, mask in enumerate(masks):
        mask_np = mask.cpu().numpy()  # Convert to NumPy array
        if mask_np[y, x]:
            return mask, idx

        # If not inside, compute the minimum distance from goal_coor to mask pixels
        mask_indices = np.column_stack(np.where(mask_np > 0))
        if mask_indices.size:
            distances = np.linalg.norm(mask_indices - np.array([y, x]), axis=1)
            min_dist = np.min(distances)
            if min_dist < min_distance:
                min_distance = min_dist
                best_mask = mask
                selected_index = idx

    return best_mask, selected_index


def visualize_cropping_box(image, cropping_box):
    # Visualize the cropping box on the image
    x1, y1, x2, y2 = cropping_box
    plt.figure()
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1,
                                      y2-y1, edgecolor='red', facecolor='none'))
    plt.title("Cropping Box Visualization")
    plt.show()


# Helper functions
def load_image_as_base64(image_path):
    with open(image_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image



def process_grasping_result(output, text):
    """
    Parses the grasping result output and extracts object ID and class name.

    Supports two formats:
    - "[ID, class name]"  -> Example: "[1, green cylinder]"
    - "[pick object, ID, class name]"  -> Example: "[pick object, 4, blue bolt]"
    """

    # Try first format: [ID, class name]
    match1 = re.search(r'\[(\d+),\s*(.+?)\]', output)
    
    # Try second format: [pick object, ID, class name]
    match2 = re.search(r'\[pick object,\s*(\d+),\s*(.+?)\]', output)

    if match1:
        object_id = int(match1.group(1))
        class_name = match1.group(2).lower()
    elif match2:
        object_id = int(match2.group(1))
        class_name = match2.group(2).lower()
    else:
        class_name = text
        return {"class_name": class_name}
    
    return {
        "selected_object_id": object_id,  # Target object ID
        "class_name": class_name  # Target object class name
    }



def create_cropping_box_from_boxes(box, image_size, margin=20):
    if len(box) == 0:
        return 0, 0, image_size[0], image_size[1]

    x1_min = float('inf')
    y1_min = float('inf')
    x2_max = float('-inf')
    y2_max = float('-inf')

    x1, y1, x2, y2 = box
    if x1 < x1_min:
        x1_min = x1
    if y1 < y1_min:
        y1_min = y1
    if x2 > x2_max:
        x2_max = x2
    if y2 > y2_max:
        y2_max = y2

    x1_min = max(0, x1_min - margin)
    y1_min = max(0, y1_min - margin)
    x2_max = min(image_size[0], x2_max + margin)
    y2_max = min(image_size[1], y2_max + margin)

    return int(x1_min), int(y1_min), int(x2_max), int(y2_max)



def get_and_process_data(cropping_box, color, depth, camera, viz):
    kernel = 0.2

    color = Image.fromarray(color)
    color = np.array(color, dtype=np.float32) / 255.0

    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    x1, y1, x2, y2 = map(int, cropping_box)
    x1_, y1_, x2_, y2_ = x1 - int((x2 - x1) * kernel) - 50, y1 - int((y2 - y1) * kernel) - 50, x2 + int(
        (x2 - x1) * kernel) + 50, y2 + int((y2 - y1) * kernel) + 50

    xmin, ymin, xmax, ymax = 0, 0, camera.width, camera.height
    dx1, dy1, dx2, dy2 = max(x1_, xmin), max(
        y1_, ymin), min(x2_, xmax), min(y2_, ymax)
    print(x1_, y1_, x2_, y2_, xmin, ymin, xmax, ymax)

    mask = np.zeros_like(depth)

    mask[dy1:dy2, dx1:dx2] = 1
    mask = (mask > 0) & (depth > 0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    print("Point cloud number", len(cloud_masked))

    num_point = 20000
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(
            len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_sampled.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_sampled.astype(np.float32))

    end_points = dict()
    cloud_sampled = torch.from_numpy(
        cloud_sampled[np.newaxis].astype(np.float32))
    cloud_sampled = cloud_sampled.to(torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))  

    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled
    if viz:
        o3d.visualization.draw_geometries([cloud])

    return end_points, cloud



### REASONING UTILS ### 
def parse_points_with_id(file_path):
    """Reads the saved sceneXXX_id.txt file and extracts Molmo IDs with GT IDs directly."""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found!")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    parsed_data = []
    molmo_to_gt_map = {}  # Direct mapping from Molmo ID → GT ID

    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) == 4:  # Ensure correct format: "Molmo_ID X Y GT_ID"
            molmo_id, x, y, gt_id = map(int, parts)
            parsed_data.append((molmo_id, x, y, gt_id))
            molmo_to_gt_map[molmo_id] = gt_id  # Store direct mapping

    return parsed_data, molmo_to_gt_map, lines


def get_prediction_result(image, path, scene_number, text):
    try:
        scene_path = os.path.join(path)
        image_path = os.path.join(scene_path, f"{scene_number}.png")
        id_file_path = os.path.join(scene_path, f"{scene_number}_id.txt")

        image_pil = image 
        # image_pil = langsamutils.load_image(str(ori_image_path))
        
        # Load image as base64
        base64_image = load_image_as_base64(image_path)
        
        # Read object ID file
        parsed_data, molmo_to_gt_map, points_with_ids = parse_points_with_id(id_file_path)

        # GPT input text
        input_text = f"Grasp {text}"

        messages = [
            {
                "role": "system",
                "content": (
               "You are a robotic system for bin picking, using a parallel gripper. I labeled all objects id in the image."

                "You have two possible actions:"

                "1. remove obstacle, object_id: This action moves the specified object out of the way so it does not interfere with grasping the desired target object. This action can only be performed if the specified object is free of obstacles (not occluded by any other object)."
                "2. pick object, object_id: This action picks up the specified object. It can only be performed if the object is free of obstacles."
                "An object is considered an obstacle if it occludes another object."

                "Task:"
                "Given a target object description as input, determine the first object that needs to be grasped to enable picking the target object. If the target object is free of obstacles, return the target object ID itself. Otherwise, identify an object that is occluding the target and is itself free of obstacles. If multiple objects could be removed, return any one valid option."

                "Output Format:"
                    "The output should only be the object ID of the first object to grasp, must formatted as: [object_id, color class_name]\n"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=713,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=0
        )

        output = response.choices[0].message.content
        result = process_grasping_result(output, text)
        

        if 'selected_object_id' not in result or result["selected_object_id"] not in [item[0] for item in parsed_data]:
            print("No object id detected by GPT, using LangSAM for segmentation...")
            input2LangSAM = result["class_name"]
            masks, boxes, phrases, logits = langsam_actor.predict(image_pil, input2LangSAM)
            
            if masks is None or masks.numel() == 0:
                raise Exception("LangSAM failed to segment the object.")

            best_index = np.argmax(logits.cpu().numpy())
            goal_mask = masks[best_index]
            goal_bbox = boxes[best_index].cpu().numpy()
        else:
            goal = result["class_name"]
            goal_id = result['selected_object_id']
            goal_coor = get_coordinates(points_with_ids, goal_id)
            masks, boxes, phrases, logits = langsam_actor.predict(image_pil, goal)

            # Selected the mask of the object we want based on the coordinates generate from molmo (if there are multi same class objects)
            goal_mask, mask_index = get_goal_mask_with_index(masks, goal_coor)
            goal_bbox = boxes[mask_index]
            goal_bbox = goal_bbox.cpu().numpy()

        return goal_mask

    except Exception as e:
        return {"error": str(e)}, {}
