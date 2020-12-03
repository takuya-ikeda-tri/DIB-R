import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

# target_num = 15
# target_num = 125
# # target_num = 350
# # target_num = 414
# base_dir = 'bottle_matrix'
# obj_file = "./tea_bottle_scan/tea_bottle_scan.obj"

target_num = 25
target_num = 350  # do debuging!!!
base_dir = 'bottle_no_box'
obj_file = "./tea_bottle_scan/tea_bottle_scan.obj"


def trans_xyz(x, y, z):
    trans = np.eye(4)
    trans[0, 3] = x
    trans[1, 3] = y
    trans[2, 3] = z
    return trans


def rot_x(theta):
    # r = np.eye(3)
    r = np.eye(4)
    r[1, 1] = np.cos(theta)
    r[1, 2] = -np.sin(theta)
    r[2, 1] = np.sin(theta)
    r[2, 2] = np.cos(theta)
    return r


def rot_y(phi):
    # r = np.eye(3)
    r = np.eye(4)
    r[0, 0] = np.cos(phi)
    r[0, 2] = np.sin(phi)
    r[2, 0] = -np.sin(phi)
    r[2, 2] = np.cos(phi)
    return r


def rot_z(psi):
    # r = np.eye(3)
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
    # pose1 is object
    trans1_pre = np.load(f"./{base_dir}/pose1/{target_num}.npy")
    # trans2_pre = np.load(f"./{base_dir}/pose2/{target_num}.npy")
    trans2_pre = np.load(f"./bottle_matrix/pose2/15.npy")

    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh1 = o3d.io.read_triangle_mesh(obj_file)
    bottle_mesh = o3d.io.read_triangle_mesh(obj_file)
    bottle1_mesh = o3d.io.read_triangle_mesh(obj_file)

    realsense = o3d.io.read_triangle_mesh(
        './realsense/Intel_RealSense_Depth_Camera_D415_mm_bigger.ply')
    realsense.paint_uniform_color([0.5, 0.5, 0.5])
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
    vive0_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vive1_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vive2_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # import cv2
    # depth = cv2.imread("./../data/rgbd/cleanser/0.png")
    # print(np.asarray(depth).dtype)
    # print(np.unique(np.asarray(depth)))
    # print(np.asarray(depth))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth)

    # plt.subplot(1, 2, 1)
    # plt.title('Redwood grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Redwood depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    mesh_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    mesh1_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # trans_obj = rot_z(np.pi / 4.0)
    trans_obj = rot_z(np.pi / 2.0)
    # trans_obj = rot_y(-np.pi / 2.0) @ rot_z(np.pi / 4.0)
    # trans_obj = rot_y(-3.0 * np.pi / 4.0) @ rot_z(np.pi / 2.0)
    # trans_obj = rot_z(np.pi / 2.0)
    # trans_obj[:3, 3] = trans_xyz(0.12, 0, 0.10)[:3, 3]
    trans_obj[:3, 3] = trans_xyz(0.12, 0, -0.1)[:3, 3]
    # mesh1.transform(trans_obj)
    # mesh_axis.transform(trans_obj)

    # rotz = rot_z(np.pi / 4.0)
    # mesh1.transform(rotz)
    # mesh_axis.transform(rotz)
    # transx = trans_xyz(0.12, 0, 0.10)
    # mesh1.transform(transx)
    # mesh_axis.transform(transx)

    # mesh1 = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # mesh2 = o3d.io.read_triangle_mesh("./tea_bottle_scan/tea_bottle_scan.obj")
    mesh_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    rotx = rot_x(np.pi / 2.0)
    rotz = rot_z(np.pi)
    trans_camera = rotx @ rotz
    # trans_camera = rotz @ rotx

    trans_camera[:3, 3] = trans_xyz(-0.04, -0.013, 0.055)[:3, 3]

    # trans_camera[:3, 3] = trans_xyz(-0.1, -0.013, 0.055)[:3, 3]
    # trans_camera[:3, 3] = trans_xyz(-0.1, -0.013, 0.1)[:3, 3]
    # mesh_camera.transform(trans_camera)

    # mesh_camera.transform(rotz)
    # mesh_camera.transform(rotx)
    # transz = trans_xyz(-0.04, -0.013, 0.055)
    # transz = trans_xyz(0, 0, 0.055)
    # mesh_camera.transform(transz)
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

    # vive1_mesh.transform(trans1)
    # vive2_mesh.transform(trans2)
    # vive1_axis.transform(trans1)
    # vive2_axis.transform(trans2)

    vive1_mesh.transform(trans1 @ rot_x(-np.pi / 2.0))
    vive2_mesh.transform(trans2 @ rot_x(-np.pi / 2.0))
    vive1_axis.transform(trans1 @ rot_x(-np.pi / 2.0))
    vive2_axis.transform(trans2 @ rot_x(-np.pi / 2.0))

    # realsense1.transform(trans2 @ rot_x(np.pi / 2.0))
    # realsense1_axis.transform(trans2 @ rot_x(np.pi / 2.0))
    realsense1.transform(
        trans2 @ rot_x(np.pi / 2.0) @ trans_xyz(0, -0.015, 0.08))
    realsense1_axis.transform(
        trans2 @ rot_x(np.pi / 2.0) @ trans_xyz(0, -0.015, 0.08))

    bottle1_mesh.transform(
        trans1 @ rot_y(np.pi) @ rot_z(-np.pi / 2.0) @ trans_xyz(0.12, 0, 0.10))

    trans_obj = trans1 @ trans_obj
    trans_camera = trans2 @ trans_camera

    mesh1.transform(trans_obj)
    mesh1_axis.transform(trans_obj)
    mesh_camera.transform(trans_camera)

    mesh.transform((np.linalg.inv(trans_camera) @ trans_obj))
    mesh_axis.transform((np.linalg.inv(trans_camera) @ trans_obj))

    vive0_trans = np.linalg.inv(trans2) @ trans1
    # vive0_trans[1, 3] = -vive0_trans[1, 3]
    # vive0_trans = rot_x(np.pi / 2.0) @ rot_z(np.pi) @ vive0_trans
    # vive0_trans = rot_z(np.pi) @ rot_x(np.pi / 2.0) @ vive0_trans
    vive0_trans = rot_z(np.pi) @ rot_y(np.pi) @ rot_x(
        np.pi / 2.0) @ vive0_trans
    vive0_mesh.transform(vive0_trans)
    vive0_axis.transform(vive0_trans)

    # mesh1.transform(trans1)
    # mesh_axis.transform(trans1)
    # mesh_camera.transform(trans2)

    # mesh.transform(trans1)
    # pcd.transform(trans2)

    # import pdb
    # pdb.set_trace()

    # trans = np.load("./../data/rgbd/cleanser/0.npy")
    # mesh.transform(trans)
    # mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd, mesh])
    # o3d.visualization.draw_geometries(
    #     [pcd, mesh, mesh1, mesh_camera, mesh_axis, origin_mesh])
    # o3d.visualization.draw_geometries([pcd, mesh, mesh_axis, origin_mesh])
    o3d.visualization.draw_geometries([
        pcd, mesh, mesh_axis, origin_mesh, vive1_mesh, vive2_mesh, vive1_axis,
        vive2_axis, vive0_mesh, vive0_axis, realsense, realsense1,
        realsense1_axis, bottle1_mesh
    ])
