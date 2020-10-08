from __future__ import print_function
from __future__ import division
import sys
sys.path.append('../utils/')
sys.path.append('./render_cuda')

sys.path.append('../utils/render')
# from renderfunc_cluster import rendermeshcolor as rendermesh

# renderers = {
#     'VertexColor': VCRender,
#     'Lambertian': Lambertian,
#     'SphericalHarmonics': SHRender,
#     'Phong': PhongRender
# }

from render_cuda.utils_render_color2 import linear
from render_cuda.rasterizer import linear_rasterizer
import numpy as np

import torch
import torch.nn as nn


def perspective_projection(points_bxpx3, faces_fx3, cameras):

    # perspective, use just one camera intrinc parameter
    camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1 = cameras
    cameratrans_rot_bx3x3 = camera_rot_bx3x3.permute(0, 2, 1)

    # follow pixel2mesh!!!
    # new_p = cam_mat * (old_p - cam_pos)
    points_bxpx3 = points_bxpx3 - camera_pos_bx3.view(-1, 1, 3)
    points_bxpx3 = torch.matmul(points_bxpx3, cameratrans_rot_bx3x3)

    camera_proj_bx1x3 = camera_proj_3x1.view(-1, 1, 3)
    xy_bxpx3 = points_bxpx3 * camera_proj_bx1x3
    xy_bxpx2 = xy_bxpx3[:, :, :2] / xy_bxpx3[:, :, 2:3]

    ##########################################################
    # 1 points
    pf0_bxfx3 = points_bxpx3[:, faces_fx3[:, 0], :]
    pf1_bxfx3 = points_bxpx3[:, faces_fx3[:, 1], :]
    pf2_bxfx3 = points_bxpx3[:, faces_fx3[:, 2], :]
    points3d_bxfx9 = torch.cat((pf0_bxfx3, pf1_bxfx3, pf2_bxfx3), dim=2)

    xy_f0 = xy_bxpx2[:, faces_fx3[:, 0], :]
    xy_f1 = xy_bxpx2[:, faces_fx3[:, 1], :]
    xy_f2 = xy_bxpx2[:, faces_fx3[:, 2], :]
    points2d_bxfx6 = torch.cat((xy_f0, xy_f1, xy_f2), dim=2)

    ######################################################
    # 2 normals
    v01_bxfx3 = pf1_bxfx3 - pf0_bxfx3
    v02_bxfx3 = pf2_bxfx3 - pf0_bxfx3

    # bs cannot be 3, if it is 3, we must specify dim
    normal_bxfx3 = torch.cross(v01_bxfx3, v02_bxfx3, dim=2)

    return points3d_bxfx9, points2d_bxfx6, normal_bxfx3


eps = 1e-15


def datanormalize(data, axis):
    # datalen = torch.sqrt(torch.sum(data**2, dim=axis, keepdim=True))
    datalen = torch.sqrt(torch.sum(data**2, dim=axis, keepdim=True) + 1e-8)
    return data / (datalen + eps)


def perspectiveprojectionnp(fovy, ratio=1.0, near=0.01, far=10.0):

    tanfov = np.tan(fovy / 2.0)
    # top = near * tanfov
    # right = ratio * top
    # mtx = [near / right, 0, 0, 0, \
    #          0, near / top, 0, 0, \
    #          0, 0, -(far+near)/(far-near), -2*far*near/(far-near), \
    #          0, 0, -1, 0]
    mtx = [[1.0 / (ratio * tanfov), 0, 0, 0], [0, 1.0 / tanfov, 0, 0],
           [
               0, 0, -(far + near) / (far - near),
               -2 * far * near / (far - near)
           ], [0, 0, -1.0, 0]]
    # return np.array(mtx, dtype=np.float32)
    return np.array([[1.0 / (ratio * tanfov)], [1.0 / tanfov], [-1]],
                    dtype=np.float32)


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
    import pdb
    pdb.set_trace()
    l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    l2[l2 == 0] = 1
    cam_mat = cam_mat / np.expand_dims(l2, 1)

    return torch.FloatTensor(cam_mat), torch.FloatTensor(cam_pos)


def compute_camera_params_torch(azimuth, elevation, distance):
    # theta = torch.deg2rad(azimuth)
    # phi = torch.deg2rad(elevation)
    theta = azimuth * np.pi / 180.0
    phi = elevation * np.pi / 180.0

    camY = distance * torch.sin(phi)
    temp = distance * torch.cos(phi)
    camX = temp * torch.cos(theta)
    camZ = temp * torch.sin(theta)
    cam_pos = torch.tensor([camX, camY, camZ])
    # array([7.87187789e-17, 1.53208889e+00, 1.28557522e+00])

    axisZ = cam_pos
    axisY = torch.tensor([0.0, 1.0, 0.0])
    axisX = torch.cross(axisY, axisZ)
    axisY = torch.cross(axisZ, axisX)
    cam_mat = torch.stack([axisX, axisY, axisZ])
    # array([[ 1.28557522e+00,  0.00000000e+00, -7.87187789e-17],
    #     [-1.20604166e-16,  1.65270364e+00, -1.96961551e+00],
    #     [ 7.87187789e-17,  1.53208889e+00,  1.28557522e+00]])
    l2 = torch.norm(cam_mat, 2, 1)
    l2[l2 == 0] = 1
    # array([1.28557522, 2.57115044, 2.        ])
    cam_mat = cam_mat / l2.unsqueeze(1)
    # array([[ 1.00000000e+00,  0.00000000e+00, -6.12323400e-17],
    #    [-4.69066938e-17,  6.42787610e-01, -7.66044443e-01],
    #    [ 3.93593894e-17,  7.66044443e-01,  6.42787610e-01]])

    return cam_mat, cam_pos

    # axisZ = cam_pos.copy()
    # axisY = np.array([0, 1, 0])
    # axisX = np.cross(axisY, axisZ)
    # axisY = np.cross(axisZ, axisX)
    # cam_mat = np.array([axisX, axisY, axisZ])

    # import pdb
    # pdb.set_trace()
    # l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    # l2[l2 == 0] = 1
    # cam_mat = cam_mat / np.expand_dims(l2, 1)

    # return torch.FloatTensor(cam_mat), torch.FloatTensor(cam_pos)


##################################################################
class VCRender(nn.Module):
    def __init__(self, height, width):
        super(VCRender, self).__init__()

        self.height = height
        self.width = width

    def forward(self, points, cameras, colors_bxpx3):
        import pdb

        ##############################################################
        # first, MVP projection in vertexshader
        points_bxpx3, faces_fx3 = points

        # camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1 = cameras

        points3d_bxfx9, points2d_bxfx6, normal_bxfx3 = \
            perspective_projection(points_bxpx3, faces_fx3, cameras)

        ################################################################
        # normal

        # decide which faces are front and which faces are back
        normalz_bxfx1 = normal_bxfx3[:, :, 2:3]
        # normalz_bxfx1 = torch.abs(normalz_bxfx1)

        # normalize normal
        normal1_bxfx3 = datanormalize(normal_bxfx3, axis=2)

        ############################################################
        # second, rasterization
        c0 = colors_bxpx3[:, faces_fx3[:, 0], :]
        c1 = colors_bxpx3[:, faces_fx3[:, 1], :]
        c2 = colors_bxpx3[:, faces_fx3[:, 2], :]
        mask = torch.ones_like(c0[:, :, :1])
        color_bxfx12 = torch.cat((c0, mask, c1, mask, c2, mask), dim=2)
        # color_bxfx9 = torch.cat((c0, c1, c2), dim=2)

        imfeat, improb_bxhxwx1 = linear_rasterizer(self.height, self.width,
                                                   points3d_bxfx9,
                                                   points2d_bxfx6,
                                                   normalz_bxfx1, color_bxfx12)
        # Error!!!
        # pdb.set_trace()
        # 1, 15000, 9
        # 1, 15000, 6
        # 1, 15000, 1
        # 1, 15000, 9
        # imfeat, improb_bxhxwx1 = linear(points3d_bxfx9, points2d_bxfx6,
        #                                 normalz_bxfx1, color_bxfx9)

        # imfeat = linear(points3d_bxfx9, points2d_bxfx6, normalz_bxfx1,
        #                 color_bxfx12)

        imrender = imfeat[:, :, :, :3]
        hardmask = imfeat[:, :, :, 3:]

        return imrender, improb_bxhxwx1, normal1_bxfx3


def texinterpolation(imtexcoord_bxhxwx2,
                     texture_bx3xthxtw,
                     filtering='nearest'):
    '''
    Note that opengl tex coord is different from pytorch coord
    ogl coord ranges from 0 to 1, y axis is from bottom to top and it supports circular mode(-0.1 is the same as 0.9)
    pytorch coord ranges from -1 to 1, y axis is from top to bottom and does not support circular 

    filtering is the same as the mode parameter for torch.nn.functional.grid_sample.
    '''

    # convert coord mode from ogl to pytorch
    imtexcoord_bxhxwx2 = torch.remainder(imtexcoord_bxhxwx2, 1.0)
    imtexcoord_bxhxwx2 = imtexcoord_bxhxwx2 * 2 - 1  # [0, 1] to [-1, 1]
    imtexcoord_bxhxwx2[:, :, :,
                       1] = -1.0 * imtexcoord_bxhxwx2[:, :, :, 1]  # reverse y

    # sample
    texcolor = torch.nn.functional.grid_sample(texture_bx3xthxtw,
                                               imtexcoord_bxhxwx2,
                                               mode=filtering)
    texcolor = texcolor.permute(0, 2, 3, 1)

    return texcolor


# Lambertian
def fragmentshader(imtexcoord_bxhxwx2,
                   texture_bx3xthxtw,
                   improb_bxhxwx1,
                   filtering='nearest'):

    # interpolation
    texcolor_bxhxwx3 = texinterpolation(imtexcoord_bxhxwx2,
                                        texture_bx3xthxtw,
                                        filtering=filtering)

    # mask
    color = texcolor_bxhxwx3 * improb_bxhxwx1

    return torch.clamp(color, 0, 1)


# Phong
def phong_fragmentshader(
    imnormal1_bxhxwx3,
    lightdirect1_bx3,
    eyedirect1_bxhxwx3,
    material_bx3x3,
    shininess_bx1,
    imtexcoord_bxhxwx2,
    texture_bx3xthxtw,
    improb_bxhxwx1,
):
    # parallel light
    lightdirect1_bx1x1x3 = lightdirect1_bx3.view(-1, 1, 1, 3)

    # lambertian
    cosTheta_bxhxwx1 = torch.sum(imnormal1_bxhxwx3 * lightdirect1_bx1x1x3,
                                 dim=3,
                                 keepdim=True)
    cosTheta_bxhxwx1 = torch.clamp(cosTheta_bxhxwx1, 0, 1)

    # specular
    reflect = -lightdirect1_bx1x1x3 + 2 * cosTheta_bxhxwx1 * imnormal1_bxhxwx3
    import pdb
    # pdb.set_trace()
    cosAlpha_bxhxwx1 = torch.sum(reflect * eyedirect1_bxhxwx3,
                                 dim=3,
                                 keepdim=True)
    cosAlpha_bxhxwx1 = torch.clamp(cosAlpha_bxhxwx1, 1e-5,
                                   1)  # should not be 0 since nan error
    cosAlpha_bxhxwx1 = torch.pow(cosAlpha_bxhxwx1,
                                 shininess_bx1.view(
                                     -1, 1, 1,
                                     1))  # shininess should be large than 0

    # simplified model
    # light color is [1, 1, 1]
    MatAmbColor_bx1x1x3 = material_bx3x3[:, 0:1, :].view(-1, 1, 1, 3)
    MatDifColor_bxhxwx3 = material_bx3x3[:, 1:2, :].view(-1, 1, 1,
                                                         3) * cosTheta_bxhxwx1
    MatSpeColor_bxhxwx3 = material_bx3x3[:, 2:3, :].view(-1, 1, 1,
                                                         3) * cosAlpha_bxhxwx1

    # tex color
    texcolor_bxhxwx3 = texinterpolation(imtexcoord_bxhxwx2, texture_bx3xthxtw)

    # ambient and diffuse rely on object color while specular doesn't
    color = (MatAmbColor_bx1x1x3 +
             MatDifColor_bxhxwx3) * texcolor_bxhxwx3 + MatSpeColor_bxhxwx3
    color = color * improb_bxhxwx1

    return torch.clamp(color, 0, 1)


class TexRender(nn.Module):
    def __init__(self, height, width, filtering='nearest'):
        super(TexRender, self).__init__()

        self.height = height
        self.width = width
        self.filtering = filtering

    def forward(self,
                points,
                cameras,
                uv_bxpx2,
                texture_bx3xthxtw,
                ft_fx3=None):

        ##############################################################
        # first, MVP projection in vertexshader
        points_bxpx3, faces_fx3 = points

        # use faces_fx3 as ft_fx3 if not given
        if ft_fx3 is None:
            ft_fx3 = faces_fx3

        # camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1 = cameras

        points3d_bxfx9, points2d_bxfx6, normal_bxfx3 = \
            perspective_projection(points_bxpx3, faces_fx3, cameras)

        ################################################################
        # normal

        # decide which faces are front and which faces are back
        normalz_bxfx1 = normal_bxfx3[:, :, 2:3]
        # normalz_bxfx1 = torch.abs(normalz_bxfx1)

        # normalize normal
        normal1_bxfx3 = datanormalize(normal_bxfx3, axis=2)

        ############################################################
        # second, rasterization
        c0 = uv_bxpx2[:, ft_fx3[:, 0], :]
        c1 = uv_bxpx2[:, ft_fx3[:, 1], :]
        c2 = uv_bxpx2[:, ft_fx3[:, 2], :]
        mask = torch.ones_like(c0[:, :, :1])
        uv_bxfx9 = torch.cat((c0, mask, c1, mask, c2, mask), dim=2)
        # uv_bxfx9 = torch.cat((c0, c1, c2), dim=2)

        imfeat, improb_bxhxwx1 = linear_rasterizer(self.height, self.width,
                                                   points3d_bxfx9,
                                                   points2d_bxfx6,
                                                   normalz_bxfx1, uv_bxfx9)
        import pdb
        # pdb.set_trace()
        # imfeat, improb_bxhxwx1 = linear(points3d_bxfx9, points2d_bxfx6,
        #                                 normalz_bxfx1, uv_bxfx9)

        imtexcoords = imfeat[:, :, :, :2]
        hardmask = imfeat[:, :, :, 2:3]

        # fragrement shader
        imrender = fragmentshader(imtexcoords,
                                  texture_bx3xthxtw,
                                  hardmask,
                                  filtering=self.filtering)

        return imrender, improb_bxhxwx1, normal1_bxfx3


class PhongRender(nn.Module):
    def __init__(self, height, width):
        super(PhongRender, self).__init__()

        self.height = height
        self.width = width

        # render with point normal or not
        self.smooth = False

    def set_smooth(self, pfmtx):
        self.smooth = True
        self.pfmtx = torch.from_numpy(pfmtx).view(1, pfmtx.shape[0],
                                                  pfmtx.shape[1]).cuda()

    def forward(self,
                points,
                cameras,
                uv_bxpx2,
                texture_bx3xthxtw,
                lightdirect_bx3,
                material_bx3x3,
                shininess_bx1,
                ft_fx3=None):

        assert lightdirect_bx3 is not None, 'When using the Phong model, light parameters must be passed'
        assert material_bx3x3 is not None, 'When using the Phong model, material parameters must be passed'
        assert shininess_bx1 is not None, 'When using the Phong model, shininess parameters must be passed'

        ##############################################################
        # first, MVP projection in vertexshader
        points_bxpx3, faces_fx3 = points

        # use faces_fx3 as ft_fx3 if not given
        if ft_fx3 is None:
            ft_fx3 = faces_fx3

        # camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1 = cameras

        points3d_bxfx9, points2d_bxfx6, normal_bxfx3 = \
            perspective_projection(points_bxpx3, faces_fx3, cameras)

        ################################################################
        # normal

        # decide which faces are front and which faces are back
        normalz_bxfx1 = normal_bxfx3[:, :, 2:3]
        # normalz_bxfx1 = torch.abs(normalz_bxfx1)

        # normalize normal
        normal1_bxfx3 = datanormalize(normal_bxfx3, axis=2)

        ####################################################
        # smooth or not
        if self.smooth:
            normal_bxpx3 = torch.matmul(
                self.pfmtx.repeat(normal_bxfx3.shape[0], 1, 1), normal_bxfx3)
            n0 = normal_bxpx3[:, faces_fx3[:, 0], :]
            n1 = normal_bxpx3[:, faces_fx3[:, 1], :]
            n2 = normal_bxpx3[:, faces_fx3[:, 2], :]
            normal_bxfx9 = torch.cat((n0, n1, n2), dim=2)
        else:
            normal_bxfx9 = normal_bxfx3.repeat(1, 1, 3)

        ############################################################
        # second, rasterization
        fnum = normal1_bxfx3.shape[1]
        bnum = normal1_bxfx3.shape[0]

        # we have uv, normal, eye to interpolate
        c0 = uv_bxpx2[:, ft_fx3[:, 0], :]
        c1 = uv_bxpx2[:, ft_fx3[:, 1], :]
        c2 = uv_bxpx2[:, ft_fx3[:, 2], :]
        mask = torch.ones_like(c0[:, :, :1])
        uv_bxfx3x3 = torch.cat((c0, mask, c1, mask, c2, mask),
                               dim=2).view(bnum, fnum, 3, -1)

        # normal & eye direction
        normal_bxfx3x3 = normal_bxfx9.view(bnum, fnum, 3, -1)
        eyedirect_bxfx9 = -points3d_bxfx9
        eyedirect_bxfx3x3 = eyedirect_bxfx9.view(-1, fnum, 3, 3)

        # TODO(taku): shoulf solve that!!!
        # feat = torch.cat((normal_bxfx3x3, eyedirect_bxfx3x3, uv_bxfx3x3),
        #                  dim=3)
        # feat = uv_bxfx3x3

        # feat = eyedirect_bxfx3x3

        feat = torch.cat((normal_bxfx3x3, eyedirect_bxfx3x3, uv_bxfx3x3),
                         dim=3)
        feat = feat.view(bnum, fnum, -1)
        import pdb

        # imfeature, improb_bxhxwx1 = linear(points3d_bxfx9, points2d_bxfx6,
        #                                    normalz_bxfx1, feat)

        imfeature, improb_bxhxwx1 = linear_rasterizer(self.height, self.width,
                                                      points3d_bxfx9,
                                                      points2d_bxfx6,
                                                      normalz_bxfx1, feat)

        # pdb.set_trace()
        ##################################################################
        imnormal = imfeature[:, :, :, :3]
        imeye = imfeature[:, :, :, 3:6]
        imtexcoords = imfeature[:, :, :, 6:8]
        immask = imfeature[:, :, :, 8:9]

        # normalize
        imnormal1 = datanormalize(imnormal, axis=3)
        lightdirect_bx3 = datanormalize(lightdirect_bx3, axis=1)
        imeye1 = datanormalize(imeye, axis=3)
        import pdb
        # pdb.set_trace()

        imrender = phong_fragmentshader(imnormal1, lightdirect_bx3, imeye1,
                                        material_bx3x3, shininess_bx1,
                                        imtexcoords, texture_bx3xthxtw, immask)

        return imrender, improb_bxhxwx1, normal1_bxfx3


renderers = {
    'VertexColor': VCRender,
    'Lambertian': TexRender,
    # 'SphericalHarmonics': SHRender,
    'Phong': PhongRender
}


class Renderer(nn.Module):
    def __init__(self,
                 height,
                 width,
                 mode='VertexColor',
                 camera_center=None,
                 camera_up=None,
                 camera_fov_y=None):
        super(Renderer, self).__init__()
        # assert mode in renderers, "Passed mode {0} must in in list of accepted modes: {1}".format(
        #     mode, renderers)
        # self.mode = mode
        self.renderer = renderers[mode](height, width)
        # self.renderer = VCRender(height, width)
        if camera_center is None:
            self.camera_center = np.array([0, 0, 0], dtype=np.float32)
        if camera_up is None:
            self.camera_up = np.array([0, 1, 0], dtype=np.float32)
        if camera_fov_y is None:
            self.camera_fov_y = 49.13434207744484 * np.pi / 180.0
        self.camera_params = None

    # def forward(self, points, *args, **kwargs):

    #     if self.camera_params is None:
    #         print(
    #             'Camera parameters have not been set, default perspective parameters of distance = 1, elevation = 30, azimuth = 0 are being used'
    #         )
    #         self.set_look_at_parameters([0], [30], [1])

    #     assert self.camera_params[0].shape[0] == points[0].shape[
    #         0], "Set camera parameters batch size must equal batch size of passed points"

    #     return self.renderer(points, self.camera_params, *args, **kwargs)

    def forward(self, points, camera_params, *args, **kwargs):

        if camera_params is None:
            print(
                'Camera parameters have not been set, default perspective parameters of distance = 1, elevation = 30, azimuth = 0 are being used'
            )
            self.set_look_at_parameters([0], [30], [1])

        assert camera_params[0].shape[0] == points[0].shape[
            0], "Set camera parameters batch size must equal batch size of passed points"

        return self.renderer(points, camera_params, *args, **kwargs)

    # def set_look_at_parameters(self, azimuth, elevation, distance):

    #     camera_projection_mtx = perspectiveprojectionnp(self.camera_fov_y, 1.0)
    #     camera_projection_mtx = torch.FloatTensor(camera_projection_mtx).cuda()

    #     camera_view_mtx = []
    #     camera_view_shift = []
    #     for a, e, d in zip(azimuth, elevation, distance):
    #         mat, pos = compute_camera_params(a, e, d)
    #         camera_view_mtx.append(mat)
    #         camera_view_shift.append(pos)
    #     camera_view_mtx = torch.stack(camera_view_mtx).cuda()
    #     camera_view_shift = torch.stack(camera_view_shift).cuda()

    #     # import pdb
    #     # pdb.set_trace()
    #     self.camera_params = [
    #         camera_view_mtx, camera_view_shift, camera_projection_mtx
    #     ]

    def set_look_at_parameters(self, azimuth, elevation, distance):

        camera_projection_mtx = perspectiveprojectionnp(self.camera_fov_y, 1.0)
        camera_projection_mtx = torch.FloatTensor(camera_projection_mtx).cuda()

        camera_view_mtx = []
        camera_view_shift = []
        for a, e, d in zip(azimuth, elevation, distance):
            mat, pos = compute_camera_params_torch(a, e, d)
            camera_view_mtx.append(mat)
            camera_view_shift.append(pos)

        # import pdb
        # pdb.set_trace()

        camera_view_mtx = torch.stack(camera_view_mtx).cuda()
        camera_view_shift = torch.stack(camera_view_shift).cuda()

        self.camera_params = [
            camera_view_mtx, camera_view_shift, camera_projection_mtx
        ]
        return [camera_view_mtx, camera_view_shift, camera_projection_mtx]

    def calc_look_at_parameters(self, azimuth, elevation, distance, device):

        camera_projection_mtx = perspectiveprojectionnp(self.camera_fov_y, 1.0)
        camera_projection_mtx = torch.FloatTensor(camera_projection_mtx).cuda()

        mat, pos = compute_camera_params_torch(azimuth, elevation, distance)
        return mat.to(device)[None], pos.to(
            device)[None], camera_projection_mtx

        camera_view_mtx = []
        camera_view_shift = []
        for a, e, d in zip(azimuth, elevation, distance):
            mat, pos = compute_camera_params_torch(a, e, d)
            camera_view_mtx.append(mat)
            camera_view_shift.append(pos)
        camera_view_mtx = torch.stack(camera_view_mtx).cuda()
        camera_view_shift = torch.stack(camera_view_shift).cuda()

        self.camera_params = [
            camera_view_mtx, camera_view_shift, camera_projection_mtx
        ]
        return [camera_view_mtx, camera_view_shift, camera_projection_mtx]

    def set_camera_parameters(self, parameters):
        self.camera_params = parameters
