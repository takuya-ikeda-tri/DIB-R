import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import json

# TODO(taku): add intrinsic matrix! camera param!!

target_num = 15
# target_num = 130
# target_num = 350
# target_num = 414
base_dir = 'bottle_matrix'
obj_file = "./tea_bottle_scan/tea_bottle_scan.obj"

# target_num = 25
# target_num = 200
# target_num = 350
# base_dir = 'bottle_no_box'
# obj_file = "./tea_bottle_scan/tea_bottle_scan.obj"

camera_file = './bottle_matrix/intrinsics.json'


def trans_xyz(x, y, z):
    trans = np.eye(4)
    trans[0, 3] = x
    trans[1, 3] = y
    trans[2, 3] = z
    return trans


def rot_x(theta):
    r = np.eye(4)
    r[1, 1] = np.cos(theta)
    r[1, 2] = -np.sin(theta)
    r[2, 1] = np.sin(theta)
    r[2, 2] = np.cos(theta)
    return r


def rot_y(phi):
    r = np.eye(4)
    r[0, 0] = np.cos(phi)
    r[0, 2] = np.sin(phi)
    r[2, 0] = -np.sin(phi)
    r[2, 2] = np.cos(phi)
    return r


def rot_z(psi):
    r = np.eye(4)
    r[0, 0] = np.cos(psi)
    r[0, 1] = -np.sin(psi)
    r[1, 0] = np.sin(psi)
    r[1, 1] = np.cos(psi)
    return r


if __name__ == "__main__":
    color = o3d.io.read_image(f"./{base_dir}/JPEGImages/{target_num}.jpg")
    depth = o3d.io.read_image(f"./{base_dir}/depth/{target_num}.png")

    # pose2 is camera
    trans2_pre = np.load(f"./bottle_matrix/pose2/15.npy")
    # trans2_pre = np.load(f"./{base_dir}/pose2/{target_num}.npy")

    # pose1 is object
    trans1_pre = np.load(f"./{base_dir}/pose1/{target_num}.npy")

    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh_local = o3d.io.read_triangle_mesh(obj_file)

    bottle_vis = o3d.io.read_triangle_mesh(obj_file)
    bottle_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    mesh1 = o3d.io.read_triangle_mesh(obj_file)
    bottle_mesh = o3d.io.read_triangle_mesh(obj_file)
    bottle1_mesh = o3d.io.read_triangle_mesh(obj_file)

    realsense = o3d.io.read_triangle_mesh(
        './realsense/Intel_RealSense_Depth_Camera_D415_mm_bigger.ply')
    realsense.paint_uniform_color([1.0, 0.5, 0.5])
    realsense.transform(rot_z(np.pi))

    realsense1 = o3d.io.read_triangle_mesh(
        './realsense/Intel_RealSense_Depth_Camera_D415_mm_bigger.ply')
    realsense1.paint_uniform_color([1.0, 0.5, 0.5])
    realsense1.transform(rot_z(np.pi))
    realsense1_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2)

    vive0_mesh = o3d.io.read_triangle_mesh(
        './HTC_VIVE_mm/HTC_Vive_Tracker_mm.obj')
    vive1_mesh = o3d.io.read_triangle_mesh(
        './HTC_VIVE_mm/HTC_Vive_Tracker_mm.obj')
    vive2_mesh = o3d.io.read_triangle_mesh(
        './HTC_VIVE_mm/HTC_Vive_Tracker_mm.obj')

    vive3_mesh = o3d.io.read_triangle_mesh(
        './HTC_VIVE_mm/HTC_Vive_Tracker_mm.obj')
    vive4_mesh = o3d.io.read_triangle_mesh(
        './HTC_VIVE_mm/HTC_Vive_Tracker_mm.obj')

    vive0_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vive1_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vive2_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    vive3_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vive4_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth)

    camera_model = o3d.camera.PinholeCameraIntrinsic()
    with open(camera_file) as file:
        camera_info = json.load(file)
    camera_model.set_intrinsics(camera_info['width'], camera_info['height'],
                                camera_info['fx'], camera_info['fy'],
                                camera_info['ppx'], camera_info['ppy'])
    # import pdb
    # pdb.set_trace()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_model)

    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #     rgbd_image,
    #     o3d.camera.PinholeCameraIntrinsic(
    #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    mesh_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    def decode_pose(trans_pre):
        trans_str = np.str(trans_pre)[4:-4].split(',')
        trans = np.eye(4)
        trans[0, 0] = float(trans_str[0])
        trans[0, 1] = float(trans_str[1])
        trans[0, 2] = float(trans_str[2])
        trans[0, 3] = float(trans_str[3][:-1])
        trans[1, 0] = float(trans_str[4][2:])
        trans[1, 1] = float(trans_str[5])
        trans[1, 2] = float(trans_str[6])
        trans[1, 3] = float(trans_str[7][:-1])
        trans[2, 0] = float(trans_str[8][2:])
        trans[2, 1] = float(trans_str[9])
        trans[2, 2] = float(trans_str[10])
        trans[2, 3] = float(trans_str[11])
        return trans

    trans1 = decode_pose(trans1_pre)
    trans2 = decode_pose(trans2_pre)
    vive1_mesh.transform(trans1 @ rot_x(-np.pi / 2.0))
    vive2_mesh.transform(trans2 @ rot_x(-np.pi / 2.0))
    vive1_axis.transform(trans1 @ rot_x(-np.pi / 2.0))
    vive2_axis.transform(trans2 @ rot_x(-np.pi / 2.0))

    vive3_mesh.transform(rot_x(np.pi) @ trans_xyz(0, -0.015, 0.08))
    vive3_axis.transform(rot_x(np.pi) @ trans_xyz(0, -0.015, 0.08))

    # realsense_trans = trans2 @ rot_x(np.pi / 2.0) @ trans_xyz(0, -0.015, 0.08)
    realsense_trans = trans2 @ rot_x(np.pi / 2.0) @ trans_xyz(
        -0.04, -0.015, 0.08)
    realsense1.transform(realsense_trans)
    realsense1_axis.transform(realsense_trans)

    bottle_trans = trans1 @ rot_y(np.pi) @ rot_z(-np.pi / 2.0) @ trans_xyz(
        0.12, 0, 0.10)
    bottle1_mesh.transform(bottle_trans)

    obj_trans = np.linalg.inv(realsense_trans) @ bottle_trans

    threshold = 0.002
    # mesh_pcd = mesh.sample_points_uniformly(number_of_points=1000)

    # aabb = mesh_pcd.get_axis_aligned_bounding_box()
    # aabb.color = (0, 0, 1)
    # obb = mesh_pcd.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)
    # crop_pcd = pcd.crop(obb)

    # trans_icp = o3d.registration.registration_icp(
    #     crop_pcd, mesh_pcd, threshold, obj_trans,
    #     o3d.registration.TransformationEstimationPointToPoint(),
    #     o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

    # mesh.transform(obj_trans)
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=2000)
    mesh_pcd.paint_uniform_color([1.0, 0, 0])

    vive_pcd = vive0_mesh.sample_points_uniformly(number_of_points=2000)

    mesh_local_pcd = mesh_local.sample_points_uniformly(number_of_points=2000)
    mesh_local_pcd.paint_uniform_color([1.0, 0, 0])

    # obb = mesh_pcd.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)
    # crop_pcd = pcd.crop(obb)
    # down_crop_pcd = crop_pcd.voxel_down_sample(voxel_size=0.005)
    # down_crop_pcd = crop_pcd.voxel_down_sample(voxel_size=0.0025)
    down_pcd = pcd.voxel_down_sample(voxel_size=0.0025)

    # o3d.visualization.draw_geometries([crop_pcd, mesh_pcd, aabb, obb])
    # o3d.visualization.draw_geometries([down_crop_pcd, mesh_pcd, aabb, obb])

    bottle_vis.transform(obj_trans)
    bottle_vis_pcd = bottle_vis.sample_points_uniformly(number_of_points=2000)
    bottle_vis_pcd.paint_uniform_color([1.0, 0, 0])
    obb = bottle_vis_pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)

    o3d.visualization.draw_geometries([down_pcd, bottle_vis_pcd, obb])

    # import pdb
    # pdb.set_trace()

    # trans_icp = o3d.registration.registration_icp(
    #     crop_pcd, mesh_pcd, threshold, np.eye(4),
    #     o3d.registration.TransformationEstimationPointToPoint(),
    #     o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

    # mesh.transform(trans_icp.transformation)

    pcd.estimate_normals()
    # crop_pcd.estimate_normals()
    # down_crop_pcd.estimate_normals()
    down_pcd.estimate_normals()
    mesh_pcd.estimate_normals()

    # trans_icp = o3d.registration.registration_icp(
    #     mesh_pcd, down_crop_pcd, threshold, np.eye(4),
    #     o3d.registration.TransformationEstimationPointToPlane(),
    #     o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

    # trans_icp = o3d.registration.registration_icp(
    #     mesh_pcd, down_pcd, threshold, np.eye(4),
    #     o3d.registration.TransformationEstimationPointToPlane(),
    #     o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

    bottle_vive_point = o3d.geometry.PointCloud()
    vive_pcd.transform(
        rot_z(-np.pi / 2.0) @ rot_x(np.pi / 2.0) @ trans_xyz(0.0, -0.1, 0.13))
    p1_load = np.asarray(mesh_pcd.points).copy()
    p2_load = np.asarray(vive_pcd.points).copy()
    p3_load = np.concatenate((p1_load, p2_load), axis=0)
    p4_load = p3_load.copy()
    p5_load = p3_load.copy()
    bottle_vive_point.points = o3d.utility.Vector3dVector(p3_load)
    bottle_vive_point.paint_uniform_color([0.0, 0, 1.0])

    bottle_vive_point_vis = o3d.geometry.PointCloud()
    bottle_vive_point_vis.points = o3d.utility.Vector3dVector(p5_load)
    bottle_vive_point_vis.paint_uniform_color([0.0, 0, 1.0])
    bottle_vive_point_vis.transform(obj_trans)
    o3d.visualization.draw_geometries([down_pcd, bottle_vive_point_vis, obb])

    bottle_vive_point_local = o3d.geometry.PointCloud()
    bottle_vive_point_local.points = o3d.utility.Vector3dVector(p4_load)
    bottle_vive_point_local.paint_uniform_color([0.0, 0, 1.0])

    bottle_vive_point.estimate_normals()
    bottle_vive_point_local.estimate_normals()

    trans_icp = o3d.registration.registration_icp(
        bottle_vive_point, down_pcd, threshold, obj_trans,
        o3d.registration.TransformationEstimationPointToPlane(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

    trans_icp_local = o3d.registration.registration_icp(
        bottle_vive_point_local.transform(obj_trans), down_pcd, threshold,
        np.eye(4), o3d.registration.TransformationEstimationPointToPlane(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

    import pdb
    pdb.set_trace()

    # trans_icp = o3d.registration.registration_icp(
    #     mesh_pcd, down_pcd, threshold, obj_trans,
    #     o3d.registration.TransformationEstimationPointToPlane(),
    #     o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

    # trans_icp_local = o3d.registration.registration_icp(
    #     mesh_local_pcd.transform(obj_trans), down_pcd, threshold, np.eye(4),
    #     o3d.registration.TransformationEstimationPointToPlane(),
    #     o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

    mesh_pcd.transform(trans_icp.transformation)
    # o3d.visualization.draw_geometries([down_pcd, mesh_pcd])

    bottle_vive_point.transform(trans_icp.transformation)
    o3d.visualization.draw_geometries([down_pcd, bottle_vive_point])

    mesh.transform(trans_icp.transformation)

    # bottle_axis.transform((trans_icp.transformation) @ obj_trans)
    bottle_axis.transform(trans_icp.transformation)
    # bottle_axis.transform(obj_trans)
    # bottle_axis.transform(trans_icp.transformation)

    vive4_mesh.transform(trans_icp_local.transformation @ obj_trans @ rot_z(
        -np.pi / 2.0) @ rot_x(np.pi / 2.0) @ trans_xyz(0.0, -0.1, 0.13))
    vive4_axis.transform(trans_icp_local.transformation @ obj_trans @ rot_z(
        -np.pi / 2.0) @ rot_x(np.pi / 2.0) @ trans_xyz(0.0, -0.1, 0.13))

    vive0_trans = np.linalg.inv(trans2) @ trans1
    vive0_trans = rot_z(np.pi) @ rot_y(np.pi) @ rot_x(
        np.pi / 2.0) @ vive0_trans
    vive0_mesh.transform(vive0_trans)
    vive0_axis.transform(vive0_trans)

    # o3d.visualization.draw_geometries([
    #     pcd, mesh, origin_mesh, vive1_mesh, vive2_mesh, vive1_axis, vive2_axis,
    #     vive0_mesh, vive0_axis, realsense, realsense1, realsense1_axis,
    #     bottle1_mesh
    # ])

    o3d.visualization.draw_geometries([
        pcd, mesh, origin_mesh, bottle_axis, realsense, vive3_mesh, vive4_mesh,
        vive3_axis, vive4_axis
    ])
