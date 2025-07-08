import torch
import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup


from models.FGC_graspnet.model.decode import pred_decode
from models.FGC_graspnet.model.FGC_graspnet import FGC_graspnet
from models.FGC_graspnet.utils.collision_detector import ModelFreeCollisionDetector


class grasp_model():
    def __init__(self, args, device, image, mask, camera_info) -> None:
        self.args = args

        # input
        self.device = device
        self.img = image
        self.mask = mask
        self.camera = camera_info

        # net parameters
        self.num_view = args.num_view
        self.checkpoint_grasp_path = args.checkpoint_grasp_path
        self.collision_thresh = args.collision_thresh
        self.viz = args.viz


    def load_grasp_net(self):
        # Init the model
        net = FGC_graspnet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                           cylinder_radius=0.05, hmin=-0.02, hmax=0.02, is_training=False, is_demo=True)
        
        net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_grasp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded FGC_GraspNet checkpoint %s (epoch: %d)"%(self.checkpoint_grasp_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net


    def check_grasp(self, gg):
        gg_top_down = GraspGroup()
        scores = []

        for grasp in gg:
            rot = grasp.rotation_matrix
            score = grasp.score

            # Target vector for top-down grasp
            target_vector = np.array([0, 0, -1])

            # Grasp approach vector
            # Assuming the grasp approach vector is the z-axis of the rotation matrix
            grasp_vector = rot @ np.array([-1, 0, 0])

            # Calculate the angle between the grasp vector and the target vector
            angle = np.arccos(
                np.clip(np.dot(grasp_vector, target_vector), -1.0, 1.0))

            # Select top-down grasp with a Z value and within 60 degrees (π/3 radians)
            if angle <= np.pi / 6:
                gg_top_down.add(grasp)
                scores.append(score)

        if len(scores) == 0:
            return GraspGroup()  # Return an empty GraspGroup if no suitable grasps found

        # Normalize scores and select the best grasps
        ref_value = np.max(scores)
        ref_min = np.min(scores)
        scores = [x - ref_min for x in scores]
        
        factor = 0.3
        if np.max(scores) > ref_value * factor:
            print('select top-down')               
            return gg_top_down
        else:
            print('no suitable grasp found')
            return GraspGroup()
    

    def pc_to_depth(self, pc, camera):
        x, y, z = pc
        xmap = x*camera.fx / z + camera.cx
        ymap = y*camera.fy / z + camera.cy

        return int(xmap), int(ymap)


    def process_masks(self, mask):
        n, h, w = mask.shape
        processed_masks = torch.zeros((h, w), dtype=mask.dtype)

        for i in range(n):
            single_mask = mask[i]
            processed_mask = single_mask
            processed_masks += processed_mask

        processed_masks = processed_masks.clamp(0, 1)
        return processed_masks


    def choose_in_mask(self, gg):
        gg_new = GraspGroup()
        self.mask = self.process_masks(self.mask)

        for grasp in gg:
            rot = grasp.rotation_matrix
            translation = grasp.translation
            if translation[-1] != 0:
                xmap, ymap = self.pc_to_depth(translation, self.camera)

                if self.mask[ymap, xmap]:
                    gg_new.add(grasp)
        return gg_new


    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)

        return gg_array, gg


    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(
            cloud, voxel_size=self.args.voxel_size)
        collision_mask = mfcdetector.detect(
            gg, approach_dist=0.05, collision_thresh=self.args.collision_thresh)
        gg = gg[~collision_mask]
        return gg


    def forward(self, end_points, cloud, path):
        grasp_net = self.load_grasp_net()
        gg_array, gg = self.get_grasps(grasp_net, end_points)

        if self.viz:
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])
        gg = self.choose_in_mask(gg)

        if self.viz:
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])

        gg = self.collision_detection(gg, np.array(cloud.points))
        gg = self.check_grasp(gg)

        gg.sort_by_score()

        gg_array = gg.grasp_group_array

        if self.viz:
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])

        o3d.io.write_point_cloud(f'{path}/cloud.ply', cloud)

        return gg, gg_array
