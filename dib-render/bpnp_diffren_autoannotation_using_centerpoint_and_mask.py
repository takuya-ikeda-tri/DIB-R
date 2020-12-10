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
from kornia.geometry import camera
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
    mat03mul = torch.tensor((ux - cx) / fx, dtype=torch.float).type_as(mat[2,
                                                                           3])
    mat13mul = torch.tensor((vy - cy) / fy, dtype=torch.float).type_as(mat[2,
                                                                           3])
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
    mat03mul = torch.tensor((ux - cx) / fx, dtype=torch.float).type_as(transz)
    mat13mul = torch.tensor((vy - cy) / fy, dtype=torch.float).type_as(transz)
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


def make_K_from_json(camera_file):
    import json
    with open(camera_file) as json_file:
        camera_json = json.load(json_file)
    cx = camera_json['ppx']
    cy = camera_json['ppy']
    fx = camera_json['fx']
    fy = camera_json['fy']
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    camera_proj_mat_np = np.array([[fx / cx], [fy / cy], [-1]])
    return K, camera_proj_mat_np


def allocentric_to_mat(azi, ele, til, ux, vy, dis, K):
    rot_xflip = torch.eye(3)
    rot_xflip[1, 1] = torch.tensor(-1)
    rot_xflip[2, 2] = torch.tensor(-1)
    uvc = torch.tensor([ux, vy, 1.0], dtype=torch.float)
    disvec = torch.tensor([0.0, 0.0, dis], dtype=torch.float)
    p = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float)
    K = K.type_as(uvc)
    q = torch.inverse(K) @ uvc
    Rcv = rot_2vector(p, q)
    Tco = Rcv @ disvec
    Rov = ((rot_y(azi) @ rot_x(-ele)) @ rot_z(til)) @ rot_xflip
    Rco = Rcv @ torch.inverse(Rov)
    mat_co = torch.eye(4)
    mat_co[:3, :3] = Rco
    mat_co[:3, 3] = Tco
    return mat_co, Rco, Tco


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


TEA_BOTTLE_SCAN_3DKPT = np.array(
    [[-3.40630002e-02, -3.40539999e-02, -1.03869997e-01],
     [-3.40630002e-02, -3.40539999e-02, 1.03869997e-01],
     [-3.40630002e-02, 3.39750014e-02, -1.03869997e-01],
     [-3.40630002e-02, 3.39750014e-02, 1.03869997e-01],
     [3.40140015e-02, -3.40539999e-02, -1.03869997e-01],
     [3.40140015e-02, -3.40539999e-02, 1.03869997e-01],
     [3.40140015e-02, 3.39750014e-02, -1.03869997e-01],
     [3.40140015e-02, 3.39750014e-02, 1.03869997e-01],
     [-2.44993716e-05, -3.94992530e-05, 0.00000000e+00]])


# TODO(taku): simplify more,
class Model(nn.Module):
    def __init__(self, renderer, depth_renderer, image_ref, sil_ref,
                 points_ref, device, verticesc, facesc, uvc, texturec,
                 tflight_bx3c, tfmatc, tfshic, camera_proj_mat_np,
                 pts2d_init_np, bpnp, K, pts3d_gt, init_pose):
        super().__init__()
        self.renderer = renderer
        self.depth_renderer = depth_renderer
        self.device = device

        # Get the reference silhouette and RGB image, points
        self.register_buffer('image_ref', image_ref)
        self.register_buffer('sil_ref', sil_ref)
        self.register_buffer('points_ref', points_ref)

        # Camera Param
        self.camera_proj_mat = torch.from_numpy(camera_proj_mat_np).to(device)

        # Initiale position parameter(BPnP)
        self.obj_2d_kpts = nn.Parameter(
            torch.from_numpy(pts2d_init_np).to(device))
        # self.init_pose = torch.from_numpy(init_pose).to(device)[None]
        self.init_pose = init_pose

        # Renderer Setting
        self.vertices = verticesc
        self.faces = facesc
        self.uv = uvc
        self.texture = texturec
        self.tflight_bx3 = tflight_bx3c
        self.tfmat = tfmatc
        self.tfshi = tfshic

        # BPnP Setting
        self.K = K
        self.bpnp = bpnp
        # self.pts3d_gt = torch.from_numpy(pts3d_gt).to(device).type_as(K)
        self.pts3d_gt = pts3d_gt

    def forward(self):
        # P_out = self.bpnp(self.obj_2d_kpts[None], self.pts3d_gt, self.K)
        P_out = self.bpnp(self.obj_2d_kpts[None], self.pts3d_gt, self.K,
                          self.init_pose)
        self.init_pose = P_out.detach()

        Rco = angle_axis_to_rotation_matrix(P_out[:, :3])[0]
        mat_co = torch.eye(4).to(self.device)
        rotation = Rco
        translation = P_out[0, 3:]
        mat_co[:3, :3] = rotation
        mat_co[:3, 3] = translation
        cam_rot, cam_trans = make_camera_mat_from_mat(mat_co, self.device)
        camera_params = [
            cam_rot[None].to(self.device), cam_trans[None].to(self.device),
            self.camera_proj_mat.type_as(self.vertices)
        ]

        # import pdb
        # pdb.set_trace()
        # Visual Alignment
        predictions, silhouette, _ = self.renderer(
            points=[self.vertices, self.faces.long()],
            camera_params=camera_params,
            uv_bxpx2=self.uv,
            texture_bx3xthxtw=self.texture,
            lightdirect_bx3=self.tflight_bx3.cuda(),
            material_bx3x3=self.tfmat.cuda(),
            shininess_bx1=self.tfshi.cuda())
        # predictions, silhouette, _ = self.renderer(
        #     points=[self.vertices, self.faces.long()],
        #     camera_params=camera_params,
        #     colors_bxpx3=colors)

        # TODO(taku): extract silhouette info too, and delete the above
        # Geometric Alignment
        xyzs = transform_pts_Rt_th(self.vertices[0], rotation.to(self.device),
                                   translation.to(self.device))[None]
        depth_predictions, _, _ = self.depth_renderer(
            points=[self.vertices, self.faces.long()],
            camera_params=camera_params,
            colors_bxpx3=xyzs)
        pre_points = points_from_depth_uv_torch_mat(
            depth_predictions[0, :, :, 2], self.K)

        loss = 0
        loss += torch.mean((predictions - self.image_ref)**2)
        loss += torch.mean((silhouette - self.sil_ref)**2)
        # Chamfer Distance
        loss += 0.1 * chamfer.chamfer_distance(self.points_ref[None],
                                               pre_points[None])[0]
        return loss, predictions, silhouette, None
        # return loss, predictions, silhouette, depth_predictions[0, :, :, 2]


def points_from_depth(depth_im, K, device):
    K = torch.from_numpy(K).type_as(depth_im).to(device)
    hw_ind = torch.nonzero(depth_im)
    coord_mat = torch.cat(
        [hw_ind[:, [1, 0]],
         torch.ones(hw_ind.shape[0])[:, None].cuda()],
        dim=1).t()
    depth_array = depth_im[(depth_im != 0).cpu().numpy()]
    result = (torch.inverse(K) @ coord_mat) * depth_array
    result = result.t()
    return result


# https://stackoverflow.com/questions/10967130/how-to-calculate-azimut-elevation-relative-to-a-camera-direction-of-view-in-3d
# https://www.mathworks.com/help/phased/ref/azel2phitheta.html
# https://math.stackexchange.com/questions/2346964/elevation-rotation-of-a-matrix-in-polar-coordinates
def main():
    USE_TEXTURE = True
    set_seed(777)
    args = parse_arguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    base_dir = 'vive_dataset/DiffRenDataset'
    obj_file = base_dir + '/' + 'models/tea_bottle_scan/tea_bottle_scan.obj'
    texture_file = base_dir + '/' + 'models/tea_bottle_scan/tea_bottle_scan.jpg'
    camera_file = base_dir + '/' + 'intrinsics.json'

    ###########################
    # Load mesh
    ###########################
    pointnp_px3, facenp_fx3, uv = loadobjtex(obj_file)
    vertices = torch.from_numpy(pointnp_px3).to(device)
    vertices = vertices.unsqueeze(0)
    faces = torch.from_numpy(facenp_fx3).to(device)

    ###########################
    # Generate texture mapping
    ###########################
    if USE_TEXTURE:
        uv = torch.from_numpy(uv).type_as(vertices)
        uv = uv.unsqueeze(0)  # 1, 6078, 2
        ###########################
        # Load texture
        ###########################
        texture = np.array(Image.open(texture_file))
        texture = torch.from_numpy(texture).cuda()
        texture = texture.float() / 255.0
        # Convert to NxCxHxW layout
        texture = texture.permute(2, 0, 1).unsqueeze(0)
        renderer_mode = 'Phong'
    else:
        ###########################
        # Generate vertex color
        ###########################
        vert_min = torch.min(vertices)
        vert_max = torch.max(vertices)
        colors = (vertices - vert_min) / (vert_max - vert_min)
        renderer_mode = 'VertexColor'

    renderer = Renderer(HEIGHT, WIDTH, mode=renderer_mode)
    vc_renderer = Renderer(HEIGHT, WIDTH, mode='VertexColor')
    K, camera_proj_mat_np = make_K_from_json(camera_file)

    ###########################
    # inference result mat
    ###########################
    init_mat_co = np.array(
        [[-0.67963279, -0.04933497, 0.73189161, -0.08272623],
         [0.68088977, 0.32879057, 0.65443554, 0.07169828],
         [-0.27292562, 0.94311337, -0.18986518, 0.63473249],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32)
    init_mat_co = torch.from_numpy(init_mat_co)
    rotation = init_mat_co[:3, :3].clone()
    translation = init_mat_co[:3, 3].clone()

    # Reference Images from Real
    color_im = Image.open(base_dir + '/' + 'rgb/15.jpg')
    depth_im = Image.open(base_dir + '/' + 'depth/15.png')
    mask_im = Image.open(base_dir + '/' + 'mask/15.png')
    # mask_im = (np.array(mask_im) / 255)[:, :, None]
    mask_im = np.array(mask_im)[:, :, None]
    color_im = np.array(color_im) * mask_im
    depth_im = np.array(depth_im) * mask_im[:, :, 0] / 1000.0  # [mm] -> [m]
    depth_im = torch.from_numpy(depth_im.astype(np.float32)).to(device)
    predictions_ref = torch.from_numpy(
        (color_im[None] / 255.0).astype(np.float32)).to(device)
    silhouete_ref = torch.from_numpy(
        (mask_im[None]).astype(np.float32)).to(device)
    points_ref = points_from_depth(depth_im, K, device)
    # points_ref = torch.from_numpy((ref_points).astype(np.float32)).to(device)

    # import pdb
    # pdb.set_trace()
    K = torch.from_numpy(K)
    '''
    ###########################
    # from allocentric to mat
    ###########################
    # TODO(taku): allocentric 4DoF Euler
    azi = torch.tensor(0.3, dtype=torch.float)
    ele = torch.tensor(0.2, dtype=torch.float)
    til = torch.tensor(0.2, dtype=torch.float)
    uloc, vloc = K[0, 2], K[1, 2]
    # uloc, vloc = 400.1305, 300.5644
    dis = 0.5
    init_mat_co, rotation, translation = allocentric_to_mat(
        azi, ele, til, uloc, vloc, dis, K)
    '''

    ###########################
    # from mat to renderer camera param representation
    ###########################
    cam_rot, cam_trans = make_camera_mat_from_mat(init_mat_co)
    camera_proj_mat = torch.FloatTensor(camera_proj_mat_np).to(device)
    camera_params = []
    camera_params.append(cam_rot[None].to(device))
    camera_params.append(cam_trans[None].to(device))
    camera_params.append(camera_proj_mat)

    ###########################
    # Prepare the K, 3d keypoints
    ###########################
    K = K.type_as(vertices)
    pts3d_gt = TEA_BOTTLE_SCAN_3DKPT

    ###########################
    # Initial value for BPnP
    ###########################
    # https://github.com/kornia/kornia/issues/317
    AngleAxis = kornia.rotation_matrix_to_angle_axis(rotation)
    Pose = torch.zeros(6, dtype=torch.float, device=device)
    Pose[:3] = AngleAxis
    Pose[3:] = translation
    Pose = Pose.reshape(1, 6)

    # Define the bpnp
    pts3d_gt = torch.from_numpy(pts3d_gt).to(device).type_as(K)
    pts2d_gt = BPnP.batch_project(Pose, pts3d_gt, K)
    bpnp_fast = BPnP.BPnP_fast.apply

    # Make initial key points
    pts2d_init_np = pts2d_gt.cpu().detach().numpy()[0]  # (9,2)
    # pts2d_init_np += np.random.rand(9, 2) * 50.0
    pts2d_init_np = pts2d_init_np.astype(np.float32)

    ###########################
    # Setting for Phong Renderer
    ###########################
    bs = len(vertices)  # vertices.shape = torch.Size([1, 6078, 3])
    # huristic phong model
    material = np.array([[0.8, 0.8, 0.8], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]],
                        dtype=np.float32).reshape(-1, 3, 3)
    tfmat = torch.from_numpy(material).repeat(bs, 1, 1)
    shininess = np.array([100], dtype=np.float32).reshape(-1, 1)
    tfshi = torch.from_numpy(shininess).repeat(bs, 1)
    lightdirect = np.array([[1.0], [1.0], [0.5]]).astype(np.float32)
    tflight = torch.from_numpy(lightdirect)
    tflight_bx3 = tflight
    '''
    # ambient color
    # material = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0., 0., 0.]],
    #                     dtype=np.float32).reshape(-1, 3, 3)
    # tfmat = torch.from_numpy(material).repeat(bs, 1, 1)
    # shininess = np.array([0], dtype=np.float32).reshape(-1, 1)
    # tfshi = torch.from_numpy(shininess).repeat(bs, 1)
    # lightdirect = np.array([[0.0], [0.0], [0.0]]).astype(np.float32)
    # tflight = torch.from_numpy(lightdirect)
    # tflight_bx3 = tflight

    # Render RGB, Silhouette Image
    # For Phong and VertexColor Setting
    if USE_TEXTURE:
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

    ###########################
    # Render depth image and calc points
    ###########################
    xyzs = transform_pts_Rt_th(
        vertices[0], rotation.to(device),
        translation.to(device))[None]  # torch.Size([1, 6078, 3])
    vc_predictions_ref, _, _ = vc_renderer(points=[vertices,
                                                   faces.long()],
                                           camera_params=camera_params,
                                           colors_bxpx3=xyzs)
    points_ref = points_from_depth_uv_torch_mat(vc_predictions_ref[0, :, :, 2],
                                                K)
    '''
    # import pdb
    # pdb.set_trace()

    model = Model(renderer, vc_renderer, predictions_ref, silhouete_ref,
                  points_ref, device, vertices, faces, uv, texture,
                  tflight_bx3, tfmat, tfshi, camera_proj_mat_np, pts2d_init_np,
                  bpnp_fast, K, pts3d_gt, Pose).to(device)
    # please reconsider the lr value
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    ###########################
    # GIF Creation Setting
    ###########################
    filename_output = "./bottle_optimization_demo.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

    # Show the init and reference RGB image
    _, image_init, silhouette_init, depth_init = model()
    plt.subplot(1, 2, 1)
    plt.imshow(image_init.detach().squeeze().cpu().numpy())
    plt.grid(False)
    plt.title("Starting position")
    plt.subplot(1, 2, 2)
    plt.imshow(model.image_ref.detach().cpu().numpy().squeeze())
    plt.grid(False)
    plt.title("Reference Image")
    plt.show()

    loop = tqdm.tqdm(range(1000))
    for i in loop:
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            loss, pre_img, pre_sil, pre_depth = model()
            loss.backward()
            optimizer.step()
            loop.set_description('Optimizing (loss %.4f)' % loss.data)
            print(loss, model.init_pose)
        if i % 10 == 0:
            image = pre_img[0].detach().cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)
    writer.close()


if __name__ == '__main__':
    main()
