# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
import argparse
import imageio
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import kornia
from pytorch3d.loss import chamfer
# from chamfer_distance import ChamferDistance as chamfer

from simple_renderer import Renderer
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
import BPnP

###########################
# Settings
###########################
MESH_SIZE = 1
HEIGHT = 640  # 256
WIDTH = 480  # 256


def angle_axis_to_rotation_matrix(angle_axis):
    rotation_matrix = kornia.quaternion_to_rotation_matrix(
        kornia.angle_axis_to_quaternion(angle_axis))
    return rotation_matrix


def loadobj(meshfile):
    v = []
    f = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if len(data) != 4 and len(data) != 7:
            continue
        if data[0] == 'v':
            # v.append([float(d) for d in data[1:]])
            v.append([float(d) for d in data[1:4]])
        if data[0] == 'f':
            data = [da.split('/')[0] for da in data]
            f.append([int(d) for d in data[1:]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    return pointnp_px3, facenp_fx3


def loadobjtex(meshfile):
    v = []
    vt = []
    f = []
    ft = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if len(data) != 4 and len(data) != 7 and len(data) != 3:
            continue
        if data[0] == 'v':
            v.append([float(d) for d in data[1:4]])
        if data[0] == 'vt':
            vt.append([float(d) for d in data[1:3]])
        if data[0] == 'f':
            data = [da.split('/') for da in data]
            f.append([int(d[0]) for d in data[1:]])
            ft.append([int(d[1]) for d in data[1:]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    ftnp_fx3 = np.array(ft, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    uvs = np.array(vt, dtype=np.float32)[:, :2]
    uvs_downsample = np.zeros((len(pointnp_px3), 2))
    for i in range(len(pointnp_px3)):
        uvs_downsample[i] = uvs[ftnp_fx3[np.where(facenp_fx3 == i)[0][0],
                                         np.where(facenp_fx3 == i)[1][0]]]
    return pointnp_px3, facenp_fx3, uvs_downsample


def compute_camera_params_torch(azimuth, elevation, distance, device):
    theta = azimuth * np.pi / 180.0
    phi = elevation * np.pi / 180.0

    camY = distance * torch.sin(phi)
    temp = distance * torch.cos(phi)
    camX = temp * torch.cos(theta)
    camZ = temp * torch.sin(theta)
    cam_pos = torch.stack([camX, camY, camZ])

    axisZ = cam_pos.clone()
    axisY = torch.tensor([0.0, 1.0, 0.0], requires_grad=True).to(device)
    axisX = torch.cross(axisY, axisZ)
    axisY = torch.cross(axisZ, axisX)
    cam_mat = torch.stack([axisX, axisY, axisZ])
    cam_mat = F.normalize(cam_mat)
    return cam_mat, cam_pos


def compute_camera_params(azimuth: float, elevation: float, distance: float):
    theta = np.deg2rad(azimuth)
    phi = np.deg2rad(elevation)
    camY = distance * np.sin(phi)
    temp = distance * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    l2[l2 == 0] = 1
    cam_mat = cam_mat / np.expand_dims(l2, 1)
    return torch.FloatTensor(cam_mat), torch.FloatTensor(cam_pos)


# symmetric over x axis
def get_spherical_coords_x(X):
    # X is N x 3
    rad = np.linalg.norm(X, axis=1)
    theta = np.arccos(X[:, 0] / rad)
    # Azimuth
    phi = np.arctan2(X[:, 2], X[:, 1])
    # Normalize both to be between [-1, 1]
    uu = (theta / np.pi) * 2 - 1
    vv = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv], 1)


def rot_x(theta, device='cpu'):
    r = torch.eye(3).to(device)
    r[1, 1] = torch.cos(theta)
    r[1, 2] = -torch.sin(theta)
    r[2, 1] = torch.sin(theta)
    r[2, 2] = torch.cos(theta)
    return r


def rot_y(phi, device='cpu'):
    r = torch.eye(3).to(device)
    r[0, 0] = torch.cos(phi)
    r[0, 2] = torch.sin(phi)
    r[2, 0] = -torch.sin(phi)
    r[2, 2] = torch.cos(phi)
    return r


def rot_z(psi, device='cpu'):
    r = torch.eye(3).to(device)
    r[0, 0] = torch.cos(psi)
    r[0, 1] = -torch.sin(psi)
    r[1, 0] = torch.sin(psi)
    r[1, 1] = torch.cos(psi)
    return r


def rot_skew(v, device):
    r = torch.zeros((3, 3)).to(device)
    r[0, 1] = -v[2]
    r[0, 2] = v[1]
    r[1, 0] = v[2]
    r[1, 2] = -v[0]
    r[2, 0] = -v[1]
    r[2, 1] = v[0]
    return r


def rot_2vector(v1, v2, device='cpu'):
    eye = torch.eye(3).to(device)
    v_cross = torch.cross(v1, v2)
    v_mul = v1 @ v2
    rx = rot_skew(v_cross, device)
    result = eye + rx + (rx @ rx) / (1 + v_mul)
    return result


def make_camera_mat_from_mat(mat, device='cpu'):
    mat[0, 3] = -1 * mat[0, 3]
    conv_mat3 = torch.eye(3).to(device)
    conv_mat3[1, 1] = -1.0
    conv_mat3[2, 2] = -1.0
    camera_r_param = conv_mat3 @ mat[:3, :3]
    tes_conv_matrix2 = torch.eye(4).to(device)
    tes_conv_matrix2[:3, :3] = torch.inverse(camera_r_param)
    camera_t_param = (tes_conv_matrix2 @ mat)[:3, 3]
    # test_conv_matrix2 is Roc? camera_t_param is Toc
    return camera_r_param, camera_t_param


def compute_camera_params_torch_no_grad(azimuth, elevation, distance, device):
    theta = azimuth * np.pi / 180.0
    phi = elevation * np.pi / 180.0

    camY = distance * torch.sin(phi)
    temp = distance * torch.cos(phi)
    camX = temp * torch.cos(theta)
    camZ = temp * torch.sin(theta)
    cam_pos = torch.stack([camX, camY, camZ])

    axisZ = cam_pos.clone()
    axisY = torch.tensor([0.0, 1.0, 0.0]).to(device)
    axisX = torch.cross(axisY, axisZ)
    axisY = torch.cross(axisZ, axisX)
    cam_mat = torch.stack([axisX, axisY, axisZ])
    cam_mat = F.normalize(cam_mat)
    return cam_mat, cam_pos


def set_seed(seed: int = 666):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Kaolin DIB-R Example')

    parser.add_argument('--mesh',
                        type=str,
                        default=os.path.join(ROOT_DIR, 'banana.obj'),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--use_texture',
                        action='store_true',
                        help='Whether to render a textured mesh')
    parser.add_argument('--texture',
                        type=str,
                        default=os.path.join(ROOT_DIR, 'texture.png'),
                        help='Specifies path to the texture to be used')
    parser.add_argument('--output_path',
                        type=str,
                        default=os.path.join(ROOT_DIR, 'results'),
                        help='Path to the output directory')

    return parser.parse_args()


# https://stackoverflow.com/questions/10967130/how-to-calculate-azimut-elevation-relative-to-a-camera-direction-of-view-in-3d
# https://www.mathworks.com/help/phased/ref/azel2phitheta.html
# https://math.stackexchange.com/questions/2346964/elevation-rotation-of-a-matrix-in-polar-coordinates
def main():
    set_seed(777)
    args = parse_arguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ###########################
    # Load mesh
    ###########################
    pointnp_px3, facenp_fx3, uv = loadobjtex('obj_000001.obj')
    vertices = torch.from_numpy(pointnp_px3).to(device)
    vertices = vertices.unsqueeze(0)
    faces = torch.from_numpy(facenp_fx3).to(device)

    ###########################
    # Generate vertex color
    ###########################
    if not args.use_texture:
        vert_min = torch.min(vertices)
        vert_max = torch.max(vertices)
        colors = (vertices - vert_min) / (vert_max - vert_min)

    ###########################
    # Generate texture mapping
    ###########################
    if args.use_texture:
        uv = torch.from_numpy(uv).type_as(vertices)
        uv = uv.unsqueeze(0)  # 1, 6078, 2

    ###########################
    # Load texture
    ###########################
    if args.use_texture:
        texture = np.array(Image.open(args.texture))
        texture = torch.from_numpy(texture).cuda()
        texture = texture.float() / 255.0
        # Convert to NxCxHxW layout
        texture = texture.permute(2, 0, 1).unsqueeze(0)

    ###########################
    # Render
    ###########################
    if args.use_texture:
        # renderer_mode = 'Lambertian'
        renderer_mode = 'Phong'
    else:
        renderer_mode = 'VertexColor'

    renderer = Renderer(HEIGHT, WIDTH, mode=renderer_mode)
    vc_renderer = Renderer(HEIGHT, WIDTH, mode='VertexColor')

    def make_trans_mat_from_quat_z(quat, transz):
        mat = torch.eye(4)
        mat[:3, :3] = kornia.quaternion_to_rotation_matrix(quat)
        mat[2, 3] = transz
        return mat

    def make_camera_mat_from_quat_uv(quat,
                                     trans,
                                     ux,
                                     vy,
                                     cx=325.2611,
                                     cy=242.04899,
                                     fx=572.4114,
                                     fy=573.57043):
        mat = make_trans_mat_from_quat_z(quat, trans)
        mat03mul = torch.tensor((ux - cx) / fx,
                                dtype=torch.float).type_as(mat[2, 3])
        mat13mul = torch.tensor((vy - cy) / fy,
                                dtype=torch.float).type_as(mat[2, 3])
        mat[0, 3] = mat[2, 3] * mat03mul * (-1)
        mat[1, 3] = mat[2, 3] * mat13mul
        conv_mat3 = torch.eye(3)
        conv_mat3[1, 1] = -1.0
        conv_mat3[2, 2] = -1.0
        camera_r_param = conv_mat3 @ mat[:3, :3]
        tes_conv_matrix2 = torch.eye(4)
        tes_conv_matrix2[:3, :3] = torch.inverse(camera_r_param)
        camera_t_param = (tes_conv_matrix2 @ mat)[:3, 3]
        # test_conv_matrix2 is Roc? camera_t_param is Toc
        return camera_r_param, camera_t_param

    def calc_trans_from_zuv(transz, ux, vy, cx, cy, fx, fy):
        mat03mul = torch.tensor((ux - cx) / fx,
                                dtype=torch.float).type_as(transz)
        mat13mul = torch.tensor((vy - cy) / fy,
                                dtype=torch.float).type_as(transz)
        transx = mat03mul * transz
        transy = mat13mul * transz
        trans = torch.stack([transx, transy, transz])
        return trans

    def points_from_depth_uv_torch_mat(depth_im, K):
        # K = torch.from_numpy(K).type_as(depth_im)
        hw_ind = torch.nonzero(depth_im)
        coord_mat = torch.cat(
            [hw_ind[:, [1, 0]],
             torch.ones(hw_ind.shape[0])[:, None].cuda()],
            dim=1).t()
        depth_array = depth_im[(depth_im != 0).cpu().numpy()]
        result = (torch.inverse(K) @ coord_mat) * depth_array
        result = result.t()
        return result

    def transform_pts_Rt_th(pts, R, t):
        """Applies a rigid transformation to 3D points.
        :param pts: nx3 tensor with 3D points.
        :param R: 3x3 rotation matrix.
        :param t: 3x1 translation vector.
        :return: nx3 tensor with transformed 3D points.
        """
        assert pts.shape[1] == 3
        if not isinstance(pts, torch.Tensor):
            pts = torch.as_tensor(pts)
        if not isinstance(R, torch.Tensor):
            R = torch.as_tensor(R).to(pts)
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t).to(pts)
        pts_t = torch.matmul(R, pts.t()) + t.view(3, 1)
        return pts_t.t()

    def calc_uv_from_xyz(translation, cx, cy, fx, fy):
        ux = translation[0] / translation[2] * fx + cx
        vy = translation[1] / translation[2] * fy + cy
        return ux, vy

    # For Debug of point cloud
    def debug_pcd(ref_points):
        # ref_points = np.array(points_from_depth_uv(depth_im, K, ux, vy))
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ref_points)
        o3d.visualization.draw_geometries([pcd])

    import json
    with open('./dataset/camera.json') as json_file:
        camera_json = json.load(json_file)
    cx = camera_json['cx']
    cy = camera_json['cy']
    fx = camera_json['fx']
    fy = camera_json['fy']
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    # Pose Setting from file
    # import yaml
    # with open('./dataset/pose/0.yaml', 'r') as yml:
    #     pose_config = yaml.load(yml)
    # translation = np.array(pose_config[0]['X_CO']['translation'],
    #                        dtype=np.float32)[:, 0]
    # translation = torch.from_numpy(translation)
    # transz = torch.tensor([translation[2]], dtype=torch.float)
    # rotation = np.array(pose_config[0]['X_CO']['rotation'], dtype=np.float32)
    # rotation = torch.from_numpy(rotation)
    # quat = kornia.rotation_matrix_to_quaternion(rotation)
    # ux, vy = calc_uv_from_xyz(translation, cx, cy, fx, fy)

    # Pose Setting by manual
    angle_axis = torch.tensor([[0.8, 0.0, 0.8]], dtype=torch.float)
    quat = kornia.angle_axis_to_quaternion(angle_axis)
    rotation = kornia.quaternion_to_rotation_matrix(quat)[0]
    translation = torch.tensor([0.0, 0.0, 0.8], dtype=torch.float)
    transz = torch.tensor([translation[2]], dtype=torch.float)
    # transz = torch.tensor([0.9], dtype=torch.float)
    ux, vy = calc_uv_from_xyz(translation, cx, cy, fx, fy)
    camera_r_param, camera_t_param = make_camera_mat_from_quat_uv(
        quat, transz, ux, vy)
    camera_proj_mat_np = np.array([[fx / cx], [fy / cy], [-1]])
    camera_proj_mat = torch.FloatTensor(camera_proj_mat_np).cuda()
    camera_params = []
    camera_params.append(camera_r_param[None].cuda())
    camera_params.append(camera_t_param[None].cuda())
    camera_params.append(camera_proj_mat)

    # TODO(taku): allocentric 3DoF
    azimuth = torch.tensor(0.0, dtype=torch.float).cuda()
    elevation = torch.tensor(60.0, dtype=torch.float).cuda()
    distance = torch.tensor(0.5, dtype=torch.float).cuda()
    cam_mat, cam_pos = compute_camera_params_torch_no_grad(
        azimuth, elevation, distance, device)
    camera_params[0] = cam_mat[None]
    camera_params[1] = cam_pos[None]

    # TODO(taku): allocentric 4DoF Euler
    azi = torch.tensor(0.3, dtype=torch.float)
    ele = torch.tensor(0.2, dtype=torch.float)
    til = torch.tensor(0.2, dtype=torch.float)
    rot_xflip = torch.eye(3)
    rot_xflip[1, 1] = torch.tensor(-1)
    rot_xflip[2, 2] = torch.tensor(-1)
    # uloc, vloc = 305.1305, 242.5644
    uloc, vloc = 400.1305, 300.5644
    uvc = torch.tensor([uloc, vloc, 1.0], dtype=torch.float)
    dis = 0.5
    disvec = torch.tensor([0.0, 0.0, dis], dtype=torch.float)
    p = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float)
    K = torch.from_numpy(K)
    K = K.type_as(uvc)
    q = torch.inverse(K) @ uvc
    Rcv = rot_2vector(p, q)
    Tco = Rcv @ disvec
    Rov = ((rot_y(azi) @ rot_x(-ele)) @ rot_z(til)) @ rot_xflip

    # angle_axis = torch.tensor([[0.8, 0.0, 0.8]], dtype=torch.float)
    # quat = kornia.angle_axis_to_quaternion(angle_axis)
    # Rov = kornia.quaternion_to_rotation_matrix(quat)[0]

    Rco = Rcv @ torch.inverse(Rov)
    mat_co = torch.eye(4)
    mat_co[:3, :3] = Rco
    mat_co[:3, 3] = Tco
    cam_rot, cam_trans = make_camera_mat_from_mat(mat_co)
    camera_params[0] = cam_rot[None].cuda()
    camera_params[1] = cam_trans[None].cuda()
    K = K.cuda()
    rotation = Rco
    translation = Tco

    # BPnP
    pts3d_gt = np.array(
        [[-0.0334153, -0.0334686, -0.104798
          ], [-0.0334153, -0.0334686, 0.104798],
         [-0.0334153, 0.0334686, -0.104798], [-0.0334153, 0.0334686, 0.104798],
         [0.0334153, -0.0334686, -0.104798], [0.0334153, -0.0334686, 0.104798],
         [0.0334153, 0.0334686, -0.104798], [0.0334153, 0.0334686, 0.104798],
         [0., 0., 0.]],
        dtype=np.float32)
    # https://github.com/kornia/kornia/issues/317
    AngleAxis = kornia.rotation_matrix_to_angle_axis(Rco)
    Pose = torch.zeros(6, dtype=torch.float, device=device)
    Pose[:3] = AngleAxis
    Pose[3:] = Tco
    Pose = Pose.reshape(1, 6)
    # tensor([ 0.1997, -0.4819, -0.0922,  0.0771,  0.0471,  0.4918], device='cuda:0')

    pts3d_gt = torch.from_numpy(pts3d_gt).to(device).type_as(K)
    pts2d_gt = BPnP.batch_project(Pose, pts3d_gt, K)
    # tensor([[[460.2317, 302.0863],
    #         [301.9629, 243.9386],
    #         [458.9205, 402.3078],
    #         [304.9771, 313.9065],
    #         [537.6598, 284.1277],
    #         [363.6392, 234.4708],
    #         [533.8432, 377.8172],
    #         [365.0552, 301.1767],
    #         [401.7071, 301.5270]]], device='cuda:0')
    bpnp = BPnP.BPnP.apply
    # Pose_check = bpnp(pts2d_gt, pts3d_gt, K)
    # import pdb
    # pdb.set_trace()

    pts2d_init_np = pts2d_gt.cpu().detach().numpy()[0]  # (9,2)
    pts2d_init_np += np.random.rand(9, 2) * 50.0
    # pts2d_init_np += np.random.rand(9, 2) * 10.0
    pts2d_init_np = pts2d_init_np.astype(np.float32)

    # Setting for Phong Renderer
    bs = len(vertices)
    material = np.array([[0.8, 0.8, 0.8], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]],
                        dtype=np.float32).reshape(-1, 3, 3)
    tfmat = torch.from_numpy(material).repeat(bs, 1, 1)
    shininess = np.array([100], dtype=np.float32).reshape(-1, 1)
    tfshi = torch.from_numpy(shininess).repeat(bs, 1)
    lightdirect = np.array([[1.0], [1.0], [0.5]]).astype(np.float32)
    tflight = torch.from_numpy(lightdirect)
    tflight_bx3 = tflight

    # Render RGB, Silhouette Image
    # For Phong and VertexColor Setting
    if args.use_texture:
        predictions_ref, silhouete_ref, _ = renderer(
            points=[vertices, faces.long()],
            camera_params=camera_params,
            uv_bxpx2=uv,
            texture_bx3xthxtw=texture,
            lightdirect_bx3=tflight_bx3.cuda(),
            material_bx3x3=tfmat.cuda(),
            shininess_bx1=tfshi.cuda())
    else:
        predictions_ref, silhouete_ref, _ = renderer(
            points=[vertices, faces.long()],
            camera_params=camera_params,
            colors_bxpx3=colors)

    # Render depth image and calc points
    # torch.Size([1, 6078, 3])
    xyzs = transform_pts_Rt_th(vertices[0], rotation.to(device),
                               translation.to(device))[None]
    vc_predictions_ref, _, _ = vc_renderer(points=[vertices,
                                                   faces.long()],
                                           camera_params=camera_params,
                                           colors_bxpx3=xyzs)
    points_ref = points_from_depth_uv_torch_mat(vc_predictions_ref[0, :, :, 2],
                                                K)

    # Reference Images from Real
    # from PIL import Image
    # color_im = Image.open('./dataset/rgb/0.jpg')
    # depth_im = Image.open('./dataset/depth/0.png')
    # mask_im = Image.open('./dataset/mask/0.png')
    # mask_im = (np.array(mask_im) / 255)[:, :, None]
    # color_im = np.array(color_im) * mask_im
    # depth_im = np.array(depth_im) * mask_im[:, :, 0] / 1000.0  # [mm] -> [m]
    # predictions_ref = torch.from_numpy(
    #     (color_im[None] / 255.0).astype(np.float32)).to(device)
    # silhouete_ref = torch.from_numpy(
    #     (mask_im[None]).astype(np.float32)).to(device)
    # points_ref = torch.from_numpy((ref_points).astype(np.float32)).to(device)

    # Show Reference RGB and Silhuette Image
    # silhouete_np = silhouete_ref.cpu().numpy()[0]
    # predictions_np = predictions_ref.cpu().numpy()[0]
    # depth_np = vc_predictions_ref[:, :, :, 2].cpu().numpy()[0]

    # Plot reference images
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # # plt.imshow(silhouete_np[..., 0])
    # plt.imshow(depth_im)
    # plt.grid(False)
    # plt.subplot(1, 2, 2)
    # # plt.imshow(predictions_np)
    # plt.imshow(depth_np)
    # plt.grid(False)
    # plt.show()

    # GIF Creation Setting
    filename_output = "./bottle_optimization_demo.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

    # TODO(taku): simplify more,
    class Model(nn.Module):
        def __init__(self, renderer, depth_renderer, image_ref, sil_ref,
                     points_ref, device, verticesc, facesc, uvc, texturec,
                     tflight_bx3c, tfmatc, tfshic):
            super().__init__()
            self.renderer = renderer
            self.depth_renderer = depth_renderer
            self.device = device

            # Get the reference silhouette and RGB image, points
            self.register_buffer('image_ref', image_ref)
            self.register_buffer('sil_ref', sil_ref)
            self.register_buffer('points_ref', points_ref)

            # Initiale pose parameter(Egonostic)
            # quaternion
            # quat = kornia.angle_axis_to_quaternion(
            #     torch.tensor([0.0, 0.0, 1.0], dtype=torch.float))
            # self.camera_quat = nn.Parameter(
            #     torch.from_numpy(
            #         np.array([[quat[0], quat[1], quat[2], quat[3]]],
            #                  dtype=np.float32)).to(device))
            # trans z
            # self.camera_trans_axis = nn.Parameter(
            #     torch.tensor([0.8], dtype=torch.float).to(device))
            # self.camera_trans_axis = torch.tensor([0.8],
            #                                       dtype=torch.float).to(device)

            # Camera Param
            camera_proj_mat_np = np.array([[fx / cx], [fy / cy], [-1]],
                                          dtype=np.float32)
            self.camera_proj_mat = torch.from_numpy(camera_proj_mat_np).to(
                device)

            # Initiale position parameter(Allocentric)
            # self.camera_position_plane = nn.Parameter(
            #     torch.from_numpy(np.array([10.0, 60.0, 2.0],
            #                               dtype=np.float32)).to(device))

            # Initiale position parameter(AllocentricFull)
            # self.azi = nn.Parameter(
            #     torch.from_numpy(np.array([0.3 + 0.1],
            #                               dtype=np.float32)).to(device))
            # self.ele = nn.Parameter(
            #     torch.from_numpy(np.array([0.2 - 0.1],
            #                               dtype=np.float32)).to(device))
            # self.tilt = nn.Parameter(
            #     torch.from_numpy(np.array([0.2 + 0.1],
            #                               dtype=np.float32)).to(device))
            # self.camera_trans = nn.Parameter(
            #     torch.from_numpy(np.array([0.5 + 0.5],
            #                               dtype=np.float32)).to(device))
            # self.camera_uvc = nn.Parameter(
            #     torch.from_numpy(
            #         np.array([400.1305 - 20.0, 300.5644 + 20.0, 1.0],
            #                  dtype=np.float32)).to(device))

            # Initiale position parameter(BPnP)
            self.obj_2d_kpts = nn.Parameter(
                torch.from_numpy(pts2d_init_np).to(device))

            # quat = kornia.angle_axis_to_quaternion(
            #     torch.tensor([0.8, 0.0, 0.9], dtype=torch.float))
            # self.camera_quat = nn.Parameter(
            #     torch.from_numpy(
            #         np.array([[quat[0], quat[1], quat[2], quat[3]]],
            #                  dtype=np.float32)).to(device))

            # Renderer Setting
            self.vertices = verticesc
            self.faces = facesc
            self.uv = uvc
            self.texture = texturec
            self.tflight_bx3 = tflight_bx3c
            self.tfmat = tfmatc
            self.tfshi = tfshic

            # ADD
            self.rot_xflip = rot_xflip.to(device)

        def forward(self):
            P_out = bpnp(self.obj_2d_kpts[None], pts3d_gt, K)
            # pts2d_pro = BPnP.batch_project(P_out, pts3d_gt, K)
            # import pdb
            # pdb.set_trace()
            # Rco = kornia.angle_axis_to_rotation_matrix(P_out[:, :3])[0]
            Rco = angle_axis_to_rotation_matrix(P_out[:, :3])[0]
            # Tco = P_out[0, 3:]

            # p = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float).to(device)
            # q = torch.inverse(K) @ self.camera_uvc
            # Rcv = rot_2vector(p, q, self.device)
            # Tco = Rcv[:, 2] * self.camera_trans
            # Rov = (
            #     (rot_y(self.azi, self.device) @ rot_x(-self.ele, self.device))
            #     @ rot_z(self.tilt, self.device)) @ self.rot_xflip

            # Rov = kornia.quaternion_to_rotation_matrix(self.camera_quat)[0]

            # Rco = Rcv @ torch.inverse(Rov)

            # import pdb
            # pdb.set_trace()

            mat_co = torch.eye(4).to(self.device)
            mat_co[:3, :3] = Rco
            mat_co[:3, 3] = P_out[0, 3:]
            # mat_co[:3, :3] = Rco
            # mat_co[:3, 3] = Tco
            cam_rot, cam_trans = make_camera_mat_from_mat(mat_co, self.device)
            camera_r_param = cam_rot.cuda()
            camera_t_param = cam_trans.cuda()

            # camera_r_param, camera_t_param = make_camera_mat_from_quat_uv(
            #     self.camera_quat, self.camera_trans_axis, ux, vy, cx, cy, fx,
            #     fy)

            # camera_r_param, camera_t_param = compute_camera_params_torch(
            #     self.camera_position_plane[0], self.camera_position_plane[1],
            #     self.camera_position_plane[2], self.device)

            camera_params = [
                camera_r_param[None].to(self.device),
                camera_t_param[None].to(self.device), self.camera_proj_mat
            ]

            # Visual Alignment
            if args.use_texture:
                # import pdb
                # pdb.set_trace()
                # vertices: torch.Size([1, 6078, 3])
                # faces: torch.Size([12152, 3])
                # self.uv: torch.Size([1, 6078, 2])
                # self.texture: torch.Size([1, 3, 2048, 2048])
                # self.tflight: torch.Size([3, 1])
                # self.tfmat: torch.Size([1, 3, 3])
                # self.tfshi: torch.Size([1, 1])
                predictions, silhouette, _ = self.renderer(
                    points=[self.vertices, self.faces.long()],
                    camera_params=camera_params,
                    uv_bxpx2=self.uv,
                    texture_bx3xthxtw=self.texture,
                    lightdirect_bx3=self.tflight_bx3.cuda(),
                    material_bx3x3=self.tfmat.cuda(),
                    shininess_bx1=self.tfshi.cuda())
                # predictions: torch.Size([1, 480, 640, 3])
                # import pdb
                # pdb.set_trace()
            else:
                predictions, silhouette, _ = self.renderer(
                    points=[vertices, faces.long()],
                    camera_params=camera_params,
                    colors_bxpx3=colors)

            # Geometric Alignment
            # rotation_xyz = kornia.quaternion_to_rotation_matrix(
            #     self.camera_quat)
            # trans_xyz = calc_trans_from_zuv(self.camera_trans_axis, ux, vy, cx,
            #                                 cy, fx, fy)
            # xyzs = transform_pts_Rt_th(self.vertices[0], rotation_xyz[0],
            #                            trans_xyz[:, 0])[None]
            # depth_predictions, _, _ = self.depth_renderer(
            #     points=[vertices, faces.long()],
            #     camera_params=camera_params,
            #     colors_bxpx3=xyzs)
            # pre_points = points_from_depth_uv_torch_mat(
            #     depth_predictions[0, :, :, 2], K)

            xyzs = transform_pts_Rt_th(self.vertices[0], Rco.to(self.device),
                                       Tco.to(self.device))[None]
            depth_predictions, _, _ = self.depth_renderer(
                points=[vertices, faces.long()],
                camera_params=camera_params,
                colors_bxpx3=xyzs)
            # pre_points = points_from_depth_uv_torch_mat(
            #     depth_predictions[0, :, :, 2], K)

            # debug_pcd(pre_points.cpu().detach().numpy())

            # Calculate the silhouette loss
            # loss = 0
            # loss = torch.sum((predictions - self.image_ref)**2)
            # loss += torch.sum((silhouette - self.sil_ref)**2)

            loss = 0
            # loss = torch.mean((predictions - self.image_ref)**2) + \
            #     torch.mean((silhouette - self.sil_ref)**2)

            # ssim = kornia.losses.SSIM(5, reduction='none')
            # loss += ssim(predictions.permute([0, 3, 1, 2]),
            #              self.image_ref.permute([0, 3, 1, 2])).mean()
            # bce = nn.BCELoss()
            # loss += bce(silhouette, self.sil_ref)

            loss += torch.mean((predictions - self.image_ref)**2)
            loss += torch.mean((silhouette - self.sil_ref)**2)
            # import pdb
            # pdb.set_trace()

            # Chamfer Distance
            # loss += 0.01 * chamfer.chamfer_distance(self.points_ref[None],
            #                                         pre_points[None])[0]
            return loss, predictions, silhouette, depth_predictions[0, :, :, 2]
            # return loss, predictions, silhouette, None
            # loss = torch.sum((predictions - self.image_ref)**2)
            # loss += torch.sum((silhouette - self.sil_ref)**2)
            # https://gist.github.com/WangZixuan/4c4cdf49ce9989175e94524afc946726
            # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/chamfer.html
            # https://discuss.pytorch.org/t/k-nearest-neighbor-in-pytorch/59695/4
            # https://discuss.pytorch.org/t/how-to-find-k-nearest-neighbor-of-a-tensor/51593

    if not args.use_texture:
        uv, texture = None, None
    model = Model(renderer, vc_renderer, predictions_ref, silhouete_ref,
                  points_ref, device, vertices, faces, uv, texture,
                  tflight_bx3, tfmat, tfshi).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

    # Show the init and reference RGB image
    _, image_init, silhouette_init, depth_init = model()
    plt.subplot(1, 2, 1)
    plt.imshow(image_init.detach().squeeze().cpu().numpy())
    # plt.imshow(depth_init.detach().cpu().numpy())
    # plt.imshow(silhouette_init.detach().squeeze().cpu().numpy())
    plt.grid(False)
    plt.title("Starting position")

    plt.subplot(1, 2, 2)
    # plt.imshow(model.image_ref.cpu().numpy().squeeze())
    plt.imshow(model.image_ref.detach().cpu().numpy().squeeze())
    # plt.imshow(model.sil_ref.cpu().numpy().squeeze())
    # import pdb
    # pdb.set_trace()
    # plt.imshow(vc_predictions_ref[0, :, :, 2].cpu().numpy().squeeze())
    plt.grid(False)
    plt.title("Reference Image")
    plt.show()

    loop = tqdm.tqdm(range(2000))
    for i in loop:
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            loss, pre_img, pre_sil, pre_depth = model()
            loss.backward()
            # loss.backward(retain_graph=True)
            optimizer.step()
            loop.set_description('Optimizing (loss %.4f)' % loss.data)
            print(loss)

        if i % 10 == 0:
            image = pre_img[0].detach().cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)
    writer.close()


if __name__ == '__main__':
    main()
