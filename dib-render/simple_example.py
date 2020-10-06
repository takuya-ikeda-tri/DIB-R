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

from renderer import Renderer
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

CAMERA_DISTANCE = 2
CAMERA_ELEVATION = 30
MESH_SIZE = 5

# HEIGHT = 256
# WIDTH = 256
HEIGHT = 512
WIDTH = 512


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
    pointnp_px3, facenp_fx3 = loadobj('banana.obj')
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


if __name__ == '__main__':
    main()
