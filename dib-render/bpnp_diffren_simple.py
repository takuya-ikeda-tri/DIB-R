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
# from pytorch3d.loss import chamfer
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
pts3d_gt = np.array(
    [[-0.0334153, -0.0334686, -0.104798], [-0.0334153, -0.0334686, 0.104798],
     [-0.0334153, 0.0334686, -0.104798], [-0.0334153, 0.0334686, 0.104798],
     [0.0334153, -0.0334686, -0.104798], [0.0334153, -0.0334686, 0.104798],
     [0.0334153, 0.0334686, -0.104798], [0.0334153, 0.0334686, 0.104798],
     [0., 0., 0.]],
    dtype=np.float32)


# Add this for kornia bug
def angle_axis_to_rotation_matrix(angle_axis):
    rotation_matrix = kornia.quaternion_to_rotation_matrix(
        kornia.angle_axis_to_quaternion(angle_axis))
    return rotation_matrix


# Load obj mesh as correct format
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


# For Depth rendering
def points_from_depth_uv_torch_mat(depth_im, K):
    hw_ind = torch.nonzero(depth_im)
    coord_mat = torch.cat(
        [hw_ind[:, [1, 0]],
         torch.ones(hw_ind.shape[0])[:, None].cuda()],
        dim=1).t()
    depth_array = depth_im[(depth_im != 0).cpu().numpy()]
    result = (torch.inverse(K) @ coord_mat) * depth_array
    result = result.t()
    return result


# For Depth rendering
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


# For allocentric represenation
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


# TODO(taku): simplify more,
class Model(nn.Module):
    def __init__(self, renderer, depth_renderer, image_ref, sil_ref,
                 points_ref, device, verticesc, facesc, uvc, texturec,
                 tflight_bx3c, tfmatc, tfshic, fx, fy, cx, cy, pts2d_init_np,
                 bpnp, K):
        super().__init__()
        self.renderer = renderer
        self.depth_renderer = depth_renderer
        self.device = device

        # Get the reference silhouette and RGB image, points
        self.register_buffer('image_ref', image_ref)
        self.register_buffer('sil_ref', sil_ref)
        self.register_buffer('points_ref', points_ref)

        # Camera Param
        camera_proj_mat_np = np.array([[fx / cx], [fy / cy], [-1]],
                                      dtype=np.float32)
        self.camera_proj_mat = torch.from_numpy(camera_proj_mat_np).to(device)

        # Initiale position parameter(BPnP)
        self.obj_2d_kpts = nn.Parameter(
            torch.from_numpy(pts2d_init_np).to(device))

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
        self.pts3d_gt = torch.from_numpy(pts3d_gt).to(device).type_as(K)

    def forward(self):
        P_out = self.bpnp(self.obj_2d_kpts[None], self.pts3d_gt, self.K)
        Rco = angle_axis_to_rotation_matrix(P_out[:, :3])[0]
        mat_co = torch.eye(4).to(self.device)
        rotation = Rco
        translation = P_out[0, 3:]
        mat_co[:3, :3] = rotation
        mat_co[:3, 3] = translation
        cam_rot, cam_trans = make_camera_mat_from_mat(mat_co, self.device)
        camera_params = [
            cam_rot[None].to(self.device), cam_trans[None].to(self.device),
            self.camera_proj_mat
        ]

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

        # Geometric Alignment
        # xyzs = transform_pts_Rt_th(self.vertices[0],
        #                            rotation.to(self.device),
        #                            translation.to(self.device))[None]
        # depth_predictions, _, _ = self.depth_renderer(
        #     points=[vertices, faces.long()],
        #     camera_params=camera_params,
        #     colors_bxpx3=xyzs)

        loss = 0
        loss += torch.mean((predictions - self.image_ref)**2)
        loss += torch.mean((silhouette - self.sil_ref)**2)
        return loss, predictions, silhouette, None
        # return loss, predictions, silhouette, depth_predictions[0, :, :, 2]


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

    ###########################
    # Set intrinsic matrix from json
    ###########################
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
    K = torch.from_numpy(K)
    mat_co, rotation, translation = allocentric_to_mat(azi, ele, til, uloc,
                                                       vloc, dis, K)

    ###########################
    # from mat to renderer camera param representation
    ###########################
    cam_rot, cam_trans = make_camera_mat_from_mat(mat_co)
    camera_proj_mat_np = np.array([[fx / cx], [fy / cy], [-1]])
    camera_proj_mat = torch.FloatTensor(camera_proj_mat_np).cuda()
    camera_params = []
    camera_params.append(cam_rot[None].cuda())
    camera_params.append(cam_trans[None].cuda())
    camera_params.append(camera_proj_mat)

    ###########################
    # Prepare the K, 3d keypoints
    ###########################
    K = K.type_as(vertices)
    pts3d_gt = np.array(
        [[-0.0334153, -0.0334686, -0.104798
          ], [-0.0334153, -0.0334686, 0.104798],
         [-0.0334153, 0.0334686, -0.104798], [-0.0334153, 0.0334686, 0.104798],
         [0.0334153, -0.0334686, -0.104798], [0.0334153, -0.0334686, 0.104798],
         [0.0334153, 0.0334686, -0.104798], [0.0334153, 0.0334686, 0.104798],
         [0., 0., 0.]],
        dtype=np.float32)

    ###########################
    # BPnP
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
    bpnp = BPnP.BPnP.apply

    # Make initial key points
    pts2d_init_np = pts2d_gt.cpu().detach().numpy()[0]  # (9,2)
    pts2d_init_np += np.random.rand(9, 2) * 50.0
    pts2d_init_np = pts2d_init_np.astype(np.float32)

    ###########################
    # Setting for Phong Renderer
    ###########################
    bs = len(vertices)  # vertices.shape = torch.Size([1, 6078, 3])
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

    ###########################
    # GIF Creation Setting
    ###########################
    filename_output = "./bottle_optimization_demo.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

    if not args.use_texture:
        uv, texture = None, None
    model = Model(renderer, vc_renderer, predictions_ref, silhouete_ref,
                  points_ref, device, vertices, faces, uv, texture,
                  tflight_bx3, tfmat, tfshi, fx, fy, cx, cy, pts2d_init_np,
                  bpnp, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

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

    loop = tqdm.tqdm(range(2000))
    for i in loop:
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            loss, pre_img, pre_sil, pre_depth = model()
            loss.backward()
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
