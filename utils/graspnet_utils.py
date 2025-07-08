import numpy as np
import open3d as o3d
from graspnetAPI.utils.utils import create_mesh_box

def get_correct_pose(g, path, viz):
    x, y, z = g.translation
    rotation = g.rotation_matrix
    depth = g.depth
    width = g.width
    
    g.score = 0.0
    
    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    center_point = np.array([x, y, z]) 

    up_x = np.mean(bottom_points[:, 0])
    up_y = np.mean(bottom_points[:, 1])
    up_z = np.mean(bottom_points[:, 2])
    up_point = np.dot(rotation, np.array([[up_x, up_y, up_z]]).T).T[0] + center_point

    left_x = np.max(left_points[:, 0]) 
    left_y = np.mean(left_points[:, 1])
    left_z = np.mean(left_points[:, 2]) 
    left_point = np.dot(rotation, np.array([[left_x, left_y, left_z]]).T).T[0] + center_point

    right_x = np.max(right_points[:, 0]) 
    right_y = np.mean(right_points[:, 1])
    right_z = np.mean(right_points[:, 2]) 
    right_point = np.dot(rotation, np.array([[right_x, right_y, right_z]]).T).T[0] + center_point

    center = (left_point + right_point) / 2 
    
    vz = (center - up_point) / np.linalg.norm(center - up_point)
    x = (center - left_point) / np.linalg.norm(center - left_point)

    y = np.cross(vz, x)
    y = y / np.linalg.norm(y)

    new_rotation = np.column_stack((x, y, vz))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

    mat = np.eye(4)
    mat[:3, :3] = new_rotation
    mat[:3, 3] = center
    frame.transform(mat)

    gg=g.to_open3d_geometry()
    gg.paint_uniform_color(np.array([0,1,0]))
    o3d.io.write_triangle_mesh(f'{path}/grasp.obj', gg)
    
    if viz:
        pcd = o3d.io.read_point_cloud(f"{path}/cloud.ply")
        o3d.visualization.draw_geometries([gg, pcd])

    return new_rotation, center, g.width


def visualize(path):
    pcd = o3d.io.read_point_cloud(f"{path}/cloud.ply")
    gg = o3d.io.read_triangle_mesh(f"{path}/grasp.obj")
    o3d.visualization.draw_geometries([gg, pcd])


    
    
    