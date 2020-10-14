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

from simple_renderer import Renderer
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

###########################
# Settings
###########################
HEIGHT = 512  # 256
WIDTH = 512  # 256
MESH_SIZE = 1
HEIGHT = 640  # 256
WIDTH = 480  # 256


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

    def make_trans_mat_from_axis(angle_axis, trans):
        mat = torch.eye(4)
        mat[:3, :3] = kornia.angle_axis_to_rotation_matrix(angle_axis)
        mat[:3, 3] = trans
        return mat

    def make_trans_mat_from_axis_z(angle_axis, transz):
        mat = torch.eye(4).type_as(angle_axis)
        mat[:3, :3] = kornia.angle_axis_to_rotation_matrix(angle_axis)
        mat[2, 3] = transz
        return mat

    def make_trans_mat_from_quat(quat, trans):
        mat = torch.eye(4)
        mat[:3, :3] = kornia.quaternion_to_rotation_matrix(quat)
        mat[:3, 3] = trans
        import pdb
        pdb.set_trace()
        return mat

    def make_trans_mat_from_quat_z(quat, transz):
        mat = torch.eye(4)
        mat[:3, :3] = kornia.quaternion_to_rotation_matrix(quat)
        mat[2, 3] = transz
        return mat

    def make_camera_mat_from_axis(angle_axis, trans):
        mat = make_trans_mat_from_axis(angle_axis, trans)
        conv_mat3 = torch.eye(3)
        conv_mat3[1, 1] = -1.0
        conv_mat3[2, 2] = -1.0
        camera_r_param = conv_mat3 @ mat[:3, :3]
        tes_conv_matrix2 = torch.eye(4)
        tes_conv_matrix2[:3, :3] = torch.inverse(camera_r_param)
        camera_t_param = (tes_conv_matrix2 @ mat)[:3, 3]
        return camera_r_param, camera_t_param

    def make_camera_mat_from_quat(quat, trans):
        # mat = make_trans_mat_from_quat(quat, trans)
        mat = make_trans_mat_from_quat_z(quat, trans)
        conv_mat3 = torch.eye(3)
        conv_mat3[1, 1] = -1.0
        conv_mat3[2, 2] = -1.0
        camera_r_param = conv_mat3 @ mat[:3, :3]
        tes_conv_matrix2 = torch.eye(4)
        tes_conv_matrix2[:3, :3] = torch.inverse(camera_r_param)
        camera_t_param = (tes_conv_matrix2 @ mat)[:3, 3]
        return camera_r_param, camera_t_param

    def xy_calc_from_zuv(ux, vy, cx, cy, fx, fy, z):
        x = (ux - cx) * z / fx
        y = (vy - cy) * z / fy
        return x, y

    def make_camera_mat_from_quat_uv(quat, trans, ux, vy,
                                     cx=325.2611, cy=242.04899,
                                     fx=572.4114, fy=573.57043):
        # mat = make_trans_mat_from_quat(quat, trans)
        mat = make_trans_mat_from_quat_z(quat, trans)
        mat03mul = torch.tensor(
            (ux-cx)/fx, dtype=torch.float).type_as(mat[2, 3])
        mat13mul = torch.tensor(
            (vy-cy)/fy, dtype=torch.float).type_as(mat[2, 3])
        mat[0, 3] = mat[2, 3] * mat03mul * (-1)
        mat[1, 3] = mat[2, 3] * mat13mul
        conv_mat3 = torch.eye(3)
        conv_mat3[1, 1] = -1.0
        conv_mat3[2, 2] = -1.0
        camera_r_param = conv_mat3 @ mat[:3, :3]
        tes_conv_matrix2 = torch.eye(4)
        tes_conv_matrix2[:3, :3] = torch.inverse(camera_r_param)
        camera_t_param = (tes_conv_matrix2 @ mat)[:3, 3]
        return camera_r_param, camera_t_param

    # from PIL import Image
    # im = Image.open('./dataset/GreenTeaRealADOffice/rgb/0.jpg')
    color_im = Image.open('./dataset/GreeTeaRealADOffice/rgb/0.jpg')
    depth_im = Image.open('./dataset/GreeTeaRealADOffice/depth/0.png')
    mask_im = Image.open('./dataset/GreeTeaRealADOffice/mask/0.png')
    import yaml
    with open('./dataset/GreeTeaRealADOffice/pose/0.yaml', 'r') as yml:
        pose_config = yaml.load(yml)
    import json
    with open('./dataset/GreeTeaRealADOffice/camera.json') as json_file:
        camera_json = json.load(json_file)

    cx = camera_json['cx']
    cy = camera_json['cy']
    fx = camera_json['fx']
    fy = camera_json['fy']

    def calc_uv_from_xyz(translation, cx, cy, fx, fy):
        ux = translation[0][0]/translation[2][0]*fx + cx
        vy = translation[1][0]/translation[2][0]*fy + cy
        return ux, vy

    # import pdb
    # pdb.set_trace()

    translation = pose_config[0]['X_CO']['translation']
    mask_im = np.array(mask_im)[:, :, None]/255
    color_im = np.array(color_im) * mask_im
    ux, vy = calc_uv_from_xyz(translation, cx, cy, fx, fy)

    # import pdb
    # pdb.set_trace()
    # color_im = Image.fromarray(color_im.astype(np.uint8))
    # color_im.show()
    # depth_im.show()
    # mask_im.show()

    # Setting for Phong Renderer
    bs = len(vertices)
    material = np.array([[0.8, 0.8, 0.8], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]],
                        dtype=np.float32).reshape(-1, 3, 3)
    tfmat = torch.from_numpy(material).repeat(bs, 1, 1)
    shininess = np.array([100], dtype=np.float32).reshape(-1, 1)
    tfshi = torch.from_numpy(shininess).repeat(bs, 1)

    # lightdirect = 2 * np.random.rand(bs, 3).astype(np.float32) - 1  # -1 - 1
    # lightdirect[:, 2] += 2
    lightdirect = np.array([[1.0], [1.0], [0.5]]).astype(np.float32)
    tflight = torch.from_numpy(lightdirect)
    tflight_bx3 = tflight

    '''
    camera_params = []
    angle_axis = torch.tensor([[0.8, 0.0, 0.8]], dtype=torch.float)
    quat = kornia.angle_axis_to_quaternion(angle_axis)
    transz = torch.tensor(
        [0.9], dtype=torch.float)
    ux = 500
    vy = 100
    cx = 325.2611
    cy = 242.04899
    fx = 572.4114
    fy = 573.57043
    camera_r_param, camera_t_param = make_camera_mat_from_quat_uv(
        quat, transz, ux, vy)
    camera_proj_mat_np = np.array([[fx / cx], [fy / cy], [-1]])
    camera_proj_mat = torch.FloatTensor(camera_proj_mat_np).cuda()
    camera_params.append(camera_r_param[None].cuda())
    camera_params.append(camera_t_param[None].cuda())
    camera_params.append(camera_proj_mat)

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

    '''
    # predictions_ref: torch.Size([1, 480, 640, 3]), torch.float32, cuda0
    # silhouete_ref: torch.Size([1, 480, 640, 1]), torch.float32, cuda0
    predictions_ref = torch.from_numpy(
        (color_im[None]/255.0).astype(np.float32)).to(device)
    silhouete_ref = torch.from_numpy(
        (mask_im[None]).astype(np.float32)).to(device)
    # Show Reference RGB and Silhuette Image
    silhouete_np = silhouete_ref.cpu().numpy()[0]
    predictions_np = predictions_ref.cpu().numpy()[0]

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouete_np[..., 0])
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(predictions_np)
    plt.grid(False)
    plt.show()

    # GIF Creation Setting
    filename_output = "./bottle_optimization_demo.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

    class Model(nn.Module):
        def __init__(self, renderer, image_ref, sil_ref, device, verticesc,
                     facesc, uvc, texturec, tflight_bx3c, tfmatc, tfshic):
            super().__init__()
            self.renderer = renderer
            self.device = device

            # Get the reference silhouette and RGB image
            self.register_buffer('image_ref', image_ref)
            self.register_buffer('sil_ref', sil_ref)

            # Initiale position parameter
            # quat = kornia.angle_axis_to_quaternion(
            #     torch.tensor([0.8-0.2, -0.2, 0.8-0.2], dtype=torch.float))
            #     torch.tensor([0.8-0.3, 0.0, 0.8-0.3], dtype=torch.float))
            # quat = kornia.angle_axis_to_quaternion(
            #     torch.tensor([0.2, 0.2, 0.2], dtype=torch.float))
            quat = kornia.angle_axis_to_quaternion(
                torch.tensor([1.0, 0.2, 1.0], dtype=torch.float))
            self.camera_quat = nn.Parameter(torch.from_numpy(
                np.array([[quat[0], quat[1], quat[2], quat[3]]],
                         dtype=np.float32)).to(device))

            self.camera_trans_axis = nn.Parameter(
                torch.tensor([0.8], dtype=torch.float).to(device))

            # cx = 325.2611
            # cy = 242.04899
            # fx = 572.4114
            # fy = 573.57043
            camera_proj_mat_np = np.array(
                [[fx / cx], [fy / cy], [-1]], dtype=np.float32)
            self.camera_proj_mat = torch.from_numpy(
                camera_proj_mat_np).to(device)

            # Renderer Setting
            self.vertices = verticesc
            self.faces = facesc
            self.uv = uvc
            self.texture = texturec
            self.tflight_bx3 = tflight_bx3c
            self.tfmat = tfmatc
            self.tfshi = tfshic

        def forward(self):
            # camera_r_param, camera_t_param = make_camera_mat_from_quat_uv(
            #     self.camera_quat, self.camera_trans_axis, ux, vy)
            camera_r_param, camera_t_param = make_camera_mat_from_quat_uv(
                self.camera_quat, self.camera_trans_axis, ux, vy, cx, cy, fx, fy)
            camera_params = [camera_r_param[None].to(self.device),
                             camera_t_param[None].to(self.device),
                             self.camera_proj_mat]

            if args.use_texture:
                predictions, silhouette, _ = self.renderer(
                    points=[self.vertices, self.faces.long()],
                    camera_params=camera_params,
                    uv_bxpx2=self.uv,
                    texture_bx3xthxtw=self.texture,
                    lightdirect_bx3=self.tflight_bx3.cuda(),
                    material_bx3x3=self.tfmat.cuda(),
                    shininess_bx1=self.tfshi.cuda())
            else:
                predictions, silhouette, _ = self.renderer(
                    points=[vertices, faces.long()],
                    camera_params=camera_params,
                    colors_bxpx3=colors)

            # Calculate the silhouette loss
            loss = 0
            # loss = torch.sum((predictions - self.image_ref)**2)
            loss += torch.sum((silhouette - self.sil_ref)**2)
            # loss = torch.mean((predictions - self.image_ref)**2)
            # loss += torch.mean((silhouette - self.sil_ref)**2)
            return loss, predictions, silhouette

    if not args.use_texture:
        uv, texture = None, None
    model = Model(renderer, predictions_ref, silhouete_ref, device, vertices,
                  faces, uv, texture, tflight_bx3, tfmat, tfshi).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Show the init and reference RGB image
    _, image_init, silhouette_init = model()
    plt.subplot(1, 2, 1)
    plt.imshow(image_init.detach().squeeze().cpu().numpy())
    # plt.imshow(silhouette_init.detach().squeeze().cpu().numpy())
    plt.grid(False)
    plt.title("Starting position")

    plt.subplot(1, 2, 2)
    plt.imshow(model.image_ref.cpu().numpy().squeeze())
    # plt.imshow(model.sil_ref.cpu().numpy().squeeze())
    plt.grid(False)
    plt.title("Reference Image")
    plt.show()

    loop = tqdm.tqdm(range(2000))
    for i in loop:
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            loss, pre_img, pre_sil = model()
            loss.backward()
            optimizer.step()
            loop.set_description('Optimizing (loss %.4f)' % loss.data)
            print(loss)
            # print(model.camera_trans_axis)

        if i % 10 == 0:
            image = pre_img[0].detach().cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)
    writer.close()


if __name__ == '__main__':
    main()
