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
import sys
sys.path.append('../utils/')
sys.path.append('./render_cuda')
sys.path.append('../utils/render')

# this is renderer part
from renderer import Renderer
# this is object loader part
from utils.utils_mesh import loadobj

import argparse
import imageio
import numpy as np
import os
import torch
import tqdm

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

###########################
# Settings
###########################
MESH_SIZE = 5
HEIGHT = 512  # 256
WIDTH = 512  # 256

import torch.nn.functional as F


def format_tensor(input,
                  dtype=torch.float32,
                  device: str = "cpu") -> torch.Tensor:
    """
    Helper function for converting a scalar value to a tensor.
    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: torch device on which the tensor should be placed.
    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device)
    if input.dim() == 0:
        input = input.view(1)
    if input.device != device:
        input = input.to(device=device)
    return input


def convert_to_tensors_and_broadcast(*args,
                                     dtype=torch.float32,
                                     device: str = "cpu"):
    """
    Helper function to handle parsing an arbitrary number of inputs (*args)
    which all need to have the same batch dimension.
    The output is a list of tensors.
    Args:
        *args: an arbitrary number of inputs
            Each of the values in `args` can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N, K_i) or (1, K_i) where K_i are
                  an arbitrary number of dimensions which can vary for each
                  value in args. In this case each input is broadcast to a
                  tensor of shape (N, K_i)
        dtype: data type to use when creating new tensors.
        device: torch device on which the tensors should be placed.
    Output:
        args: A list of tensors of shape (N, K_i)
    """
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype, device) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % sizes
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N, ) + (-1, ) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))

    if len(args) == 1:
        args_Nd = args_Nd[0]  # Return the first element

    return args_Nd


def look_at_rotation(camera_position,
                     at=((0, 0, 0), ),
                     up=((0, 1, 0), ),
                     device: str = "cpu") -> torch.Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.
    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)
    The vectors are broadcast against each other so they all have shape (N, 3).
    Returns:
        R: (N, 3, 3) batched rotation matrices
    """
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast(camera_position,
                                                        at,
                                                        up,
                                                        device=device)

    camera_position, at, up = broadcasted_args
    for t, n in zip([camera_position, at, up],
                    ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)

    # import pdb
    # pdb.set_trace()

    is_close = torch.isclose(x_axis, torch.tensor(0.0).to(device),
                             atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]),
                  dim=1)
    return R.transpose(1, 2)


# def compute_camera_params_torch(azimuth, elevation, distance):
#     # theta = torch.deg2rad(azimuth)
#     # phi = torch.deg2rad(elevation)
#     theta = azimuth * np.pi / 180.0
#     phi = elevation * np.pi / 180.0

#     camY = distance * torch.sin(phi)
#     temp = distance * torch.cos(phi)
#     camX = temp * torch.cos(theta)
#     camZ = temp * torch.sin(theta)
#     cam_pos = torch.tensor([camX, camY, camZ])
#     # array([7.87187789e-17, 1.53208889e+00, 1.28557522e+00])

#     axisZ = cam_pos
#     axisY = torch.tensor([0.0, 1.0, 0.0])
#     axisX = torch.cross(axisY, axisZ)
#     axisY = torch.cross(axisZ, axisX)
#     cam_mat = torch.stack([axisX, axisY, axisZ])
#     # array([[ 1.28557522e+00,  0.00000000e+00, -7.87187789e-17],
#     #     [-1.20604166e-16,  1.65270364e+00, -1.96961551e+00],
#     #     [ 7.87187789e-17,  1.53208889e+00,  1.28557522e+00]])
#     l2 = torch.norm(cam_mat, 2, 1)
#     l2[l2 == 0] = 1
#     # array([1.28557522, 2.57115044, 2.        ])
#     cam_mat = cam_mat / l2.unsqueeze(1)
#     # array([[ 1.00000000e+00,  0.00000000e+00, -6.12323400e-17],
#     #    [-4.69066938e-17,  6.42787610e-01, -7.66044443e-01],
#     #    [ 3.93593894e-17,  7.66044443e-01,  6.42787610e-01]])

#     return cam_mat, cam_pos


def compute_camera_params_torch(azimuth, elevation, distance, device):
    # import pdb
    # pdb.set_trace()
    # theta = torch.deg2rad(azimuth)
    # phi = torch.deg2rad(elevation)
    theta = azimuth * np.pi / 180.0
    phi = elevation * np.pi / 180.0

    camY = distance * torch.sin(phi)
    temp = distance * torch.cos(phi)
    camX = temp * torch.cos(theta)
    camZ = temp * torch.sin(theta)
    cam_pos = torch.stack([camX, camY, camZ])
    # cam_pos = torch.tensor([camX, camY, camZ], requires_grad=True)
    # array([7.87187789e-17, 1.53208889e+00, 1.28557522e+00])

    axisZ = cam_pos.clone()
    axisY = torch.tensor([0.0, 1.0, 0.0], requires_grad=True).to(device)
    axisX = torch.cross(axisY, axisZ)
    axisY = torch.cross(axisZ, axisX)
    cam_mat = torch.stack([axisX, axisY, axisZ])
    cam_mat = F.normalize(cam_mat)
    # array([[ 1.28557522e+00,  0.00000000e+00, -7.87187789e-17],
    #     [-1.20604166e-16,  1.65270364e+00, -1.96961551e+00],
    #     [ 7.87187789e-17,  1.53208889e+00,  1.28557522e+00]])
    '''
    l2 = torch.norm(cam_mat, 2, 1)
    l2[l2 == 0] = 1

    # tensor([[  2.5981,   0.0000,  -1.5000],
    #         [ -7.7942,   9.0000, -13.5000],
    #         [  1.5000,   5.1962,   2.5981]], device='cuda:0',
    # tensor([ 3., 18.,  6.], device='cuda:0', grad_fn=<IndexPutBackward>)
    import pdb
    pdb.set_trace()

    # array([1.28557522, 2.57115044, 2.        ])
    cam_mat = cam_mat / l2.unsqueeze(1)
    # array([[ 1.00000000e+00,  0.00000000e+00, -6.12323400e-17],
    #    [-4.69066938e-17,  6.42787610e-01, -7.66044443e-01],
    #    [ 3.93593894e-17,  7.66044443e-01,  6.42787610e-01]])

    # tensor([[ 0.8660,  0.0000, -0.5000],
    #         [-0.4330,  0.5000, -0.7500],
    #         [ 0.2500,  0.8660,  0.4330]], device='cuda:0', grad_fn=<DivBackward0>)
    '''
    return cam_mat, cam_pos


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


def main():
    args = parse_arguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ###########################
    # Load mesh
    ###########################
    # pointnp_px3, facenp_fx3 = loadobj('sphere.obj')
    # pointnp_px3 /= 3.0
    # pointnp_px3, facenp_fx3 = loadobj('banana.obj')
    pointnp_px3, facenp_fx3 = loadobj('obj_000001.obj')
    pointnp_px3 = pointnp_px3[:, [1, 2, 0]]
    # import pdb
    # pdb.set_trace()
    vertices = torch.from_numpy(pointnp_px3).to(device)
    faces = torch.from_numpy(facenp_fx3).to(device)

    # Expand such that batch size = 1
    vertices = vertices.unsqueeze(0)

    ###########################
    # Normalize mesh position
    ###########################
    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.
    vertices = (vertices - vertices_middle) * MESH_SIZE

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
        uv = get_spherical_coords_x(vertices[0].cpu().numpy())
        uv = torch.from_numpy(uv).cuda()
        # Expand such that batch size = 1
        uv = uv.unsqueeze(0)

    ###########################
    # Load texture
    ###########################
    if args.use_texture:
        # Load image as numpy array
        texture = np.array(Image.open(args.texture))
        # Convert numpy array to PyTorch tensor
        texture = torch.from_numpy(texture).cuda()
        # Convert from [0, 255] to [0, 1]
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

    azimuth = torch.as_tensor(0.0)
    elevation = torch.as_tensor(50.0)
    camera_distance = torch.as_tensor(2.0)
    camera_params = renderer.set_look_at_parameters([90 - azimuth],
                                                    [elevation],
                                                    [camera_distance])

    # CAMERA_DISTANCE = 2
    # CAMERA_ELEVATION = 30
    # renderer.set_look_at_parameters([90 - azimuth], [CAMERA_ELEVATION],
    #                                 [CAMERA_DISTANCE])

    bs = len(vertices)
    material = np.array([[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]],
                        dtype=np.float32).reshape(-1, 3, 3)
    tfmat = torch.from_numpy(material).repeat(bs, 1, 1)

    shininess = np.array([100], dtype=np.float32).reshape(-1, 1)
    tfshi = torch.from_numpy(shininess).repeat(bs, 1)

    lightdirect = 2 * np.random.rand(bs, 3).astype(np.float32) - 1
    lightdirect[:, 2] += 2
    tflight = torch.from_numpy(lightdirect)
    tflight_bx3 = tflight

    # predictions_ref, silhouete_ref, _ = renderer(
    #     points=[vertices, faces.long()],
    #     uv_bxpx2=uv,
    #     texture_bx3xthxtw=texture,
    #     lightdirect_bx3=tflight_bx3.cuda(),
    #     material_bx3x3=tfmat.cuda(),
    #     shininess_bx1=tfshi.cuda())

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

    silhouete_np = silhouete_ref.cpu().numpy()[0]
    predictions_np = predictions_ref.cpu().numpy()[0]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(
        silhouete_np[..., 0])  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(predictions_np)
    plt.grid(False)
    plt.show()

    import torch.nn as nn
    from torch.autograd import Variable
    from skimage import img_as_ubyte
    filename_output = "./bottle_optimization_demo.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

    class Model(nn.Module):
        def __init__(self, renderer, image_ref, sil_ref, device, verticesc,
                     facesc, uvc, texturec, tflight_bx3c, tfmatc, tfshic):
            super().__init__()
            self.renderer = renderer

            # Get the silhouette of the reference RGB image by finding all non-white pixel values.
            self.register_buffer('image_ref', image_ref)
            self.register_buffer('sil_ref', sil_ref)

            self.camera_position_plane = nn.Parameter(
                torch.from_numpy(np.array([30.0, 60.0, 5.0],
                                          dtype=np.float32)).to(device))

            # Create an optimizable parameter for the x, y, z position of the camera.
            # self.camera_position = nn.Parameter(
            #     torch.from_numpy(np.array([3.0, 6.9, +2.5],
            #                               dtype=np.float32)).to(device))
            # self.camera_position = nn.Parameter(
            #     torch.from_numpy(np.array([3.0, -6.9, 2.5],
            #                               dtype=np.float32)).to(device))

            # self.azimuth = Variable(torch.tensor(30.0)).to(device)
            # self.elevation = Variable(torch.tensor(60.0)).to(device)
            # self.distance = Variable(torch.tensor(6.0)).to(device)
            self.vertices = verticesc
            self.faces = facesc
            self.uv = uvc
            self.texture = texturec
            self.tflight_bx3 = tflight_bx3c
            self.tfmat = tfmatc
            self.tfshi = tfshic
            self.device = device

        def forward(self):
            cam_mat, cam_pos = compute_camera_params_torch(
                90 - self.camera_position_plane[0],
                self.camera_position_plane[1], self.camera_position_plane[2],
                self.device)

            _, _, camera_mtx = self.renderer.calc_look_at_parameters(
                90 - self.camera_position_plane[0],
                self.camera_position_plane[1], self.camera_position_plane[2],
                self.device)

            # import pdb
            # pdb.set_trace()

            camera_params = [
                cam_mat[None].to(self.device), cam_pos[None].to(self.device),
                camera_mtx
            ]

            # import pdb
            # pdb.set_trace()

            # camera_params = self.renderer.set_look_at_parameters(
            #     [90 - self.camera_position_plane[0]],
            #     [self.camera_position_plane[1]],
            #     [self.camera_position_plane[2]])

            # self.renderer.set_look_at_parameters(
            #     [90 - self.camera_position[0]], [self.camera_position[1]],
            #     [self.camera_position[2]])

            # _, _, camera_mtx = self.renderer.calc_look_at_parameters(
            #     90 - self.camera_position[0], self.camera_position[1],
            #     self.camera_position[2], self.device)

            # R = look_at_rotation(self.camera_position[None, :],
            #                      device=self.device)  # (1, 3, 3)
            # T = -torch.bmm(R.transpose(
            #     1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)
            # T += torch.tensor([[1.5, 5.19, 0]]).type_as(T)

            # import pdb
            # pdb.set_trace()
            # camera_params = [R, T, camera_mtx]

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
            # loss = ((predictions - self.image_ref)**2).mean()
            # loss = 0.3 * torch.sum((predictions - self.image_ref)**2)
            loss = torch.sum((predictions - self.image_ref)**2)
            loss += torch.sum((silhouette - self.sil_ref)**2)
            # loss = torch.sum((silhouette - self.sil_ref)**2)
            # loss = 0.05 * torch.sum((predictions - self.image_ref)**2)
            # import pdb
            # pdb.set_trace()
            # loss = ((predictions - self.image_ref)**2).mean()
            # loss = Variable(loss, requires_grad=True)
            return loss, predictions, silhouette

    if not args.use_texture:
        uv, texture = None, None
    model = Model(renderer, predictions_ref, silhouete_ref, device, vertices,
                  faces, uv, texture, tflight_bx3, tfmat, tfshi).to(device)
    # model = Model(renderer, silhouete_ref, device, vertices, faces, uv,
    #               texture, tflight_bx3, tfmat, tfshi).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

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

    # loop = tqdm.tqdm(range(500))
    loop = tqdm.tqdm(range(2000))
    for i in loop:
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            loss, pre_img, pre_sil = model()
            # for param in model.parameters():
            #     print(param)
            # import pdb
            # pdb.set_trace()
            # loss.requires_grad = True
            loss.backward()
            optimizer.step()

            loop.set_description('Optimizing (loss %.4f)' % loss.data)
            print(loss)
            # if loss.item() < 200:
            #     break

        if i % 10 == 0:
            image = pre_img[0].detach().cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)
    writer.close()
    '''
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description('Drawing')
    os.makedirs(args.output_path, exist_ok=True)
    writer = imageio.get_writer(os.path.join(args.output_path, 'example.gif'),
                                mode='I')
    bs = len(vertices)
    lightdirect = 2 * np.random.rand(bs, 3).astype(np.float32) - 1
    for azimuth in loop:
        renderer.set_look_at_parameters([90 - azimuth], [CAMERA_ELEVATION],
                                        [CAMERA_DISTANCE])
        # vertices: 1, 7500, 3
        # faces: 15000, 3
        # colors: 1, 7500, 3
        if args.use_texture:
            # predictions, _, _ = renderer(points=[vertices,
            #                                      faces.long()],
            #                              uv_bxpx2=uv,
            #                              texture_bx3xthxtw=texture)

            bs = len(vertices)
            material = np.array(
                [[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]],
                dtype=np.float32).reshape(-1, 3, 3)
            shininess = np.array([100], dtype=np.float32).reshape(-1, 1)
            tfmat = torch.from_numpy(material).repeat(bs, 1, 1)
            tfshi = torch.from_numpy(shininess).repeat(bs, 1)

            # lightdirect = 2 * np.random.rand(bs, 3).astype(np.float32) - 1
            lightdirect[:, 2] += 2
            tflight = torch.from_numpy(lightdirect)
            tflight_bx3 = tflight

            predictions, _, _ = renderer(points=[vertices,
                                                 faces.long()],
                                         uv_bxpx2=uv,
                                         texture_bx3xthxtw=texture,
                                         lightdirect_bx3=tflight_bx3.cuda(),
                                         material_bx3x3=tfmat.cuda(),
                                         shininess_bx1=tfshi.cuda())

        else:
            predictions, _, _ = renderer(points=[vertices,
                                                 faces.long()],
                                         colors_bxpx3=colors)

        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image * 255).astype(np.uint8))
    writer.close()
    '''


if __name__ == '__main__':
    main()
