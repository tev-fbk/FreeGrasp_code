import os
import cv2
import json

from grasp_model import grasp_model
from models.langsam import langsamutils

from models.FGC_graspnet.utils.data_utils import CameraInfo

from molmo_eval import process_and_send_to_gpt

from utils.utils import *
from utils.config import *
from utils.graspnet_utils import get_correct_pose



def compute_grasp_pose(path, camera_info):
    parser = argparse.ArgumentParser('RUN an experiment with real data', parents=[get_args_parser()])
    args = parser.parse_args()
    
    try:
        path = str(path)
        if not path.startswith('/'):
            path = '/' + path
        image_path = os.path.join(path, "image.png")
        depth_path = os.path.join(path, "depth.npz")
        text_path = os.path.join(path, "task.txt")
        
        prompt = "Point out the objects in the red rectangle on the table."

        # Use molmo pre-process (label number) the image
        base64_labeled_image, labeled_text = process_and_send_to_gpt(image_path, prompt, path)
        
        img_ori = cv2.imread(image_path)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        with open(text_path, 'r') as file:
            text = file.read()

        depth_ori = np.load(depth_path)
        depth_ori = depth_ori['depth']

        image_pil = langsamutils.load_image(image_path)

        input_text = text
        messages = [
            {
                "role": "system",
                "content": (
               "You are a robotic system for bin picking, using a parallel gripper. I labeled all objects id in the image.\n"

                "You have two possible actions:"

                "1. remove obstacle, object_id: This action moves the specified object out of the way so it does not interfere with grasping the desired target object. This action can only be performed if the specified object is free of obstacles (not occluded by any other object).\n"
                "2. pick object, object_id: This action picks up the specified object. It can only be performed if the object is free of obstacles."
                "An object is considered an obstacle if it occludes another object.\n"

                "Task:"
                "Given a target object description as input, determine the first object that needs to be grasped to enable picking the target object. If the target object is free of obstacles, return the target object ID itself. Otherwise, identify an object that is occluding the target and is itself free of obstacles. If multiple objects could be removed, return any one valid option.\n"
                
                
                "Output Format:"
                    "The output should only be the object ID of the first object to grasp, must formatted as: [object_id, color class_name]\n"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_labeled_image}"}}
                ]
            }
        ]

        result = {
            "selected_object": None,
            "cropping_box": None,
            "objects": []
        }

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=713,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        output = response.choices[0].message.content
        result = process_grasping_result(output, text)
        
        # Goal object predicted by GPT-4o (resoning part)
        goal = result['class_name']
        goal_id = result['selected_object_id']
        goal_coor = get_coordinates(labeled_text, goal_id)
        
        with open(f"{path}/log.txt", "a") as file:
            file.write(f"I have to remove the object with id = {str(goal_id)}, named {goal}")

        masks, boxes, phrases, logits = langsam_actor.predict(image_pil, goal)

        # Selected the mask of the object we want based on the coordinates generate from molmo (if there are multi same class objects)
        goal_mask, mask_index = get_goal_mask_with_index(masks, goal_coor)
        goal_bbox = boxes[mask_index]
        goal_bbox = goal_bbox.cpu().numpy()

        cropping_box = create_cropping_box_from_boxes(
            goal_bbox, (img_ori.shape[1], img_ori.shape[0]))
        
        goal_mask = goal_mask.unsqueeze(0)

        if args.viz:
            visualize_cropping_box(img_ori, cropping_box)

        langsam_actor.save(masks, boxes, phrases, logits, image_pil, path, viz=args.viz)
        
        endpoint, pcd = get_and_process_data(
            cropping_box, img_ori, depth_ori, camera_info, viz=args.viz)
        
        grasp_net = grasp_model(args=args, device="cuda",
                                image=img_ori, mask=goal_mask, camera_info=camera_info)
            
        gg, _ = grasp_net.forward(endpoint, pcd, path)
        
        if len(gg) == 0:
            data = {}
        else:
            R, t, w = get_correct_pose(gg[0], path, args.viz)
            
            data = {
                'translation': t.tolist(),
                'rotation': R.tolist(),
                'width': w
            }
            
            save_path = os.path.join(path, "grasp_pose.json")
            with open(save_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

        return data   
    except Exception as e:
        return e



if __name__ == "__main__":
    # NOTE: to change if you want to try with your own images
    camera =  CameraInfo(width=1280, height=720, fx=912.481, fy=910.785, cx=644.943, cy=353.497, scale=1000.0)

    images = [
        "/media/tev/data/code/FreeGrasp_code/data/real_examples/hard/1",
        "/media/tev/data/code/FreeGrasp_code/data/real_examples/hard/2",
        "/media/tev/data/code/FreeGrasp_code/data/real_examples/hard/3",
        ]
    
    for i in images:
        print(compute_grasp_pose(i, camera))
    
        

