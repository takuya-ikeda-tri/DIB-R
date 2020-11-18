import torch
import torch.utils.data as torch_data
import pytorch_lightning as pl
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
from PIL import Image
from simple_renderer import Renderer

params = {
    'obj_path': './obj_000001.obj',
    'text_path': './real_tea_bottle.png',
    'height': 128,
    'width': 128
}


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


def rot_skew(v, device='cpu'):
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


def build_transform(is_training):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


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


class P2PDataset(torch_data.Dataset):
    def __init__(self, transform, img_num=1000, device='cuda'):
        self.img_num = img_num
        self.transform = transform

    def __len__(self):
        return self.img_num

    def read_data(self, idx):
        img = np.array(
            Image.open(
                './pix2pose_data/rgb/{}.png'.format(idx)).convert("RGB"))
        nocs = np.array(
            Image.open(
                './pix2pose_data/nocs/{}.png'.format(idx)).convert("RGB"))
        pose = np.load('./pix2pose_data/pose/{}.npy'.format(idx))
        return img, nocs, pose

    def __getitem__(self, idx):
        img, nocs, pose = self.read_data(idx)
        target = {}
        # TODO(taku): please check it
        target['nocs'] = TF.to_tensor(nocs)
        target['pose'] = torch.from_numpy(pose)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


'''
class P2PDataset(torch_data.Dataset):
    def __init__(self, img_ids=1000, device='cuda'):
        pointnp_px3, facenp_fx3, uv = loadobjtex(params['obj_path'])
        vertices = torch.from_numpy(pointnp_px3)
        self.vertices = vertices.unsqueeze(0).to(device)
        self.faces = torch.from_numpy(facenp_fx3).to(device)
        vert_min = torch.min(self.vertices)
        vert_max = torch.max(self.vertices)
        self.colors = (self.vertices - vert_min) / (vert_max - vert_min)
        uv = torch.from_numpy(uv).type_as(vertices)
        self.uv = uv.unsqueeze(0).to(device)  # 1, 6078, 2
        texture = np.array(Image.open(params['text_path']))
        texture = torch.from_numpy(texture)
        texture = texture.float() / 255.0
        # Convert to NxCxHxW layout
        self.texture = texture.permute(2, 0, 1).unsqueeze(0).to(device)
        self.phong_renderer = Renderer(params['height'],
                                       params['width'],
                                       mode='Phong')
        self.vc_renderer = Renderer(params['height'],
                                    params['width'],
                                    mode='VertexColor')

        # Phong Settings
        bs = 1
        material = np.array(
            [[0.8, 0.8, 0.8], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]],
            dtype=np.float32).reshape(-1, 3, 3)
        self.tfmat = torch.from_numpy(material).repeat(bs, 1, 1).to(device)
        shininess = np.array([100], dtype=np.float32).reshape(-1, 1)
        self.tfshi = torch.from_numpy(shininess).repeat(bs, 1).to(device)
        lightdirect = np.array([[1.0], [1.0], [0.5]]).astype(np.float32)
        self.tflight_bx3 = torch.from_numpy(lightdirect).to(device)

        # length of dataset
        self.img_ids = img_ids

        # camera params
        K = np.eye(3).astype(np.float32)
        fx = 200.0
        fy = 200.0
        cx = params['width'] / 2.0
        cy = params['height'] / 2.0
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        camera_proj = np.array([[fx / cx], [fy / cy], [-1]]).astype(np.float32)
        self.K = torch.from_numpy(K).to(device)
        self.camera_proj = torch.from_numpy(camera_proj).to(device)

        # transform
        # self.transform = transform

    def __len__(self):
        return self.img_ids

    def __getitem__(self, idx):
        camera_params, mat_co = self.gen_camera_params()
        target = {}
        # colors: [6078, 3], camera_params: 3, faces: [12152, 3], vertices: [1, 6078, 3]
        nocs, nocs_mask, _ = self.vc_renderer(
            points=[self.vertices, self.faces.long()],
            camera_params=camera_params,
            colors_bxpx3=self.colors)
        target['nocs'] = nocs[0].permute(2, 0, 1)
        target['nocs_mask'] = nocs_mask[0].permute(2, 0, 1)
        target['pose'] = mat_co

        image, _, _ = self.phong_renderer(
            points=[self.vertices, self.faces.long()],
            camera_params=camera_params,
            uv_bxpx2=self.uv,
            texture_bx3xthxtw=self.texture,
            lightdirect_bx3=self.tflight_bx3,
            material_bx3x3=self.tfmat,
            shininess_bx1=self.tfshi)

        # if self.transform is not None:
        #     image = self.transform(image)

        return image[0].permute(2, 0, 1), target
        # return nocs[0].permute(2, 0, 1), target

    def gen_camera_params(self, dis=0.5, device='cuda'):
        azi = torch.rand(1, dtype=torch.float)[0] * 3.14 * 2
        ele = torch.rand(1, dtype=torch.float)[0] * 3.14 * 2
        til = torch.rand(1, dtype=torch.float)[0] * 3.14 * 2
        uloc, vloc = params['width'] / 2.0, params['height'] / 2.0
        mat_co, rotation, translation = allocentric_to_mat(
            azi, ele, til, uloc, vloc, dis, self.K)
        cam_rot, cam_trans = make_camera_mat_from_mat(mat_co)
        camera_params = []
        camera_params.append(cam_rot[None].to(device))
        camera_params.append(cam_trans[None].to(device))
        camera_params.append(self.camera_proj)
        return camera_params, mat_co
'''

# class P2PDataModule(pl.LightningDataModule):
#     def __init__(self):
#         super().__init__()


def ten2num(input_tensor, ttype=torch.FloatTensor):
    return input_tensor.type(ttype).numpy()


if __name__ == "__main__":
    transform = build_transform(True)
    dataset = P2PDataset(transform)
    # dataloader = torch_data.DataLoader(dataset, batch_size=2)

    dataloader = torch_data.DataLoader(dataset, batch_size=1)
    # for i in range(1000):
    #     image, target = next(iter(dataloader))
    #     image = transforms.ToPILImage()(image[0].cpu()).convert("RGB")
    #     nocs = transforms.ToPILImage()(target['nocs'][0].cpu()).convert("RGB")
    #     pose = ten2num(target['pose'])
    #     np.save('./pix2pose_data/pose/{}.npy'.format(i), pose)
    #     image.save('./pix2pose_data/rgb/{}.png'.format(i))
    #     nocs.save('./pix2pose_data/nocs/{}.png'.format(i))

    batch = next(iter(dataloader))
    image, target = batch
    # import pdb
    # pdb.set_trace()
    import matplotlib.pyplot as plt
    plt.imshow(ten2num(image[0].permute(1, 2, 0)))
    plt.show()
    plt.imshow(ten2num(target['nocs'][0].permute(1, 2, 0)))
    plt.show()
    # plt.imshow(ten2num(target['nocs_mask'][0, 0]))
    # plt.show()

    # plt.imshow(ten2num(image[1].permute(1, 2, 0)))
    # plt.show()
    # plt.imshow(ten2num(target['nocs'][1].permute(1, 2, 0)))
    # plt.show()
    # plt.imshow(ten2num(target['nocs_mask'][1, 0]))
    # plt.show()
    print(target['pose'][0])
    # print(target['pose'][1])
