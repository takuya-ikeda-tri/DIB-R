import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.utils.data import DataLoader
from torchsummary import summary

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
import wandb

from simple_renderer import Renderer
from utils.object_pose_estimation_center_utils import draw_keypoints
import numpy as np
from PIL import Image
import BPnP
import kornia
import datetime

params = {
    'obj_path': './obj_000001.obj',
    'text_path': './real_tea_bottle.png',
    'input_size': 128,  # 256,
    'width': 128,
    'height': 128,
    'batch_size': 1,
    'num_workers': 8,
    'checkpoint': './lightning_logs/good_for_domainA/epoch=200.ckpt'
}

kpts3d = np.array([[-0.0334153, -0.0334686, -0.104798],
                   [-0.0334153, -0.0334686, 0.104798],
                   [-0.0334153, 0.0334686, -0.104798],
                   [-0.0334153, 0.0334686, 0.104798],
                   [0.0334153, -0.0334686, -0.104798],
                   [0.0334153, -0.0334686, 0.104798],
                   [0.0334153, 0.0334686, -0.104798],
                   [0.0334153, 0.0334686, 0.104798]])

K = np.eye(3).astype(np.float32)
K[0, 0] = 200.0
K[1, 1] = 200.0
K[0, 2] = params['width'] / 2.0
K[1, 2] = params['height'] / 2.0

camera_proj_mat_np = np.array(
    [[200.0 / params['width'] * 2.0], [200.0 / params['height'] * 2.0], [-1]],
    dtype=np.float32)

# bpnp = BPnP.BPnP.apply
bpnp = BPnP.BPnP_fast.apply


def project(points_3d, intrinsics, pose):
    points_3d = np.dot(points_3d, pose[:, :3].T) + pose[:, 3:].T
    points_3d = np.dot(points_3d, intrinsics.T)
    points_2d = points_3d[:, :2] / points_3d[:, 2:]
    return points_2d


def angle_axis_to_rotation_matrix(angle_axis):
    rotation_matrix = kornia.quaternion_to_rotation_matrix(
        kornia.angle_axis_to_quaternion(angle_axis))
    return rotation_matrix


# def angle_axis_to_rotation_matrix(angle_axis):
#     # rotation_matrix = kornia.quaternion_to_rotation_matrix(
#     #     kornia.angle_axis_to_quaternion(angle_axis))
#     rotation_matrix = kornia.angle_axis_to_rotation_matrix(angle_axis)
#     return rotation_matrix


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


def build_transform(is_training):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


class VGGPoseTrainDataset(torch_data.Dataset):
    def __init__(self, transform, img_num=10000, device='cuda'):
        self.img_num = img_num
        self.transform = transform

    def __len__(self):
        return self.img_num

    def read_data(self, idx):
        img = np.array(
            Image.open('./pix2pose_domainA_data/rgb/{}.png'.format(
                idx)).convert("RGB"))
        nocs = np.array(
            Image.open('./pix2pose_domainA_data/nocs/{}.png'.format(
                idx)).convert("RGB"))
        pose = np.load('./pix2pose_domainA_data/pose/{}.npy'.format(idx))
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


class VGGPoseValDataset(torch_data.Dataset):
    def __init__(self, transform, img_num=10000, device='cuda'):
        self.img_num = img_num
        self.transform = transform

    def __len__(self):
        return self.img_num

    def read_data(self, idx):
        img = np.array(
            Image.open('./pix2pose_domainB_data/rgb/{}.png'.format(
                idx)).convert("RGB"))
        nocs = np.array(
            Image.open('./pix2pose_domainB_data/nocs/{}.png'.format(
                idx)).convert("RGB"))
        pose = np.load('./pix2pose_domainB_data/pose/{}.npy'.format(idx))
        return img, nocs, pose

    def __getitem__(self, idx):
        img, nocs, pose = self.read_data(idx)
        target = {}
        # TODO(taku): please check ilightning_logs/good_for_domainA/epoch\=200.ckpt
        # TODO(taku): please check it
        target['nocs'] = TF.to_tensor(nocs)
        target['pose'] = torch.from_numpy(pose)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


class VGGPoseDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = VGGPoseTrainDataset
        self.val_dataset = VGGPoseValDataset

    def setup(self, stage):
        train_transform = build_transform(True)
        val_transform = build_transform(False)
        if stage == 'fit':
            train_data = self.train_dataset(train_transform)
            val_data = self.val_dataset(val_transform)

            # indices = torch.randperm(len(train_data)).tolist()
            # split_th = int(len(train_data) * params['split_ratio'])
            # self.train_data = torch.utils.data.Subset(train_data, [5])
            # self.train_data = torch.utils.data.Subset(val_data, [11])
            # self.val_data = torch.utils.data.Subset(val_data, [11])
            self.train_data = torch.utils.data.Subset(val_data, [5])
            self.val_data = torch.utils.data.Subset(val_data, [5])
            # self.val_data = torch.utils.data.Subset(train_data, [5])
        if stage == 'test':
            pass

    def train_dataloader(self):
        train_data = DataLoader(self.train_data,
                                batch_size=params['batch_size'],
                                shuffle=True,
                                num_workers=params['num_workers'],
                                drop_last=True)
        return train_data

    def val_dataloader(self):
        val_data = DataLoader(self.val_data,
                              batch_size=1,
                              shuffle=False,
                              num_workers=params['num_workers'],
                              drop_last=True)
        return val_data


def ten2num(input_tensor, ttype=torch.FloatTensor):
    return input_tensor.type(ttype).numpy()


class PoseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.n = 8
        self.model = torchvision.models.vgg11()
        self.model.classifier = torch.nn.Linear(25088, self.n * 2)
        self.pts3d = torch.from_numpy(kpts3d)
        self.K = torch.from_numpy(K)
        self.ini_pose = torch.zeros(1, 6)
        self.ini_pose[0, 5] = 99  # 1.0

        # For renderer
        self.renderer = Renderer(params['height'],
                                 params['width'],
                                 mode='Phong')
        # camera_proj_mat_np = np.array([[200.0 / params['width'] * 2.0],
        #                                [200.0 / params['height'] * 2.0], [-1]])
        self.camera_proj_mat = torch.from_numpy(camera_proj_mat_np)

        # vertices, facenp, uv
        pointnp_px3, facenp_fx3, uv = loadobjtex('obj_000001.obj')
        vertices = torch.from_numpy(pointnp_px3)
        self.vertices = vertices.unsqueeze(0)
        self.faces = torch.from_numpy(facenp_fx3)
        uv = torch.from_numpy(uv).type_as(vertices)
        self.uv = uv.unsqueeze(0)  # 1, 6078, 2

        # Texture
        texture = np.array(Image.open(params['text_path']))
        texture = torch.from_numpy(texture)
        texture = texture.float() / 255.0
        self.texture = texture.permute(2, 0, 1).unsqueeze(0)

        # Phong Setting
        bs = 1
        material = np.array(
            [[0.7, 0.7, 0.7], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]],
            dtype=np.float32).reshape(-1, 3, 3)
        # material = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0., 0., 0.]],
        #                     dtype=np.float32).reshape(-1, 3, 3)
        self.tfmat = torch.from_numpy(material).repeat(bs, 1, 1)
        shininess = np.array([100], dtype=np.float32).reshape(-1, 1)
        # shininess = np.array([0], dtype=np.float32).reshape(-1, 1)
        self.tfshi = torch.from_numpy(shininess).repeat(bs, 1)
        lightdirect = np.array([[1.0], [1.0], [0.5]]).astype(np.float32)
        # lightdirect = np.array([[0.0], [0.0], [0.0]]).astype(np.float32)
        self.tflight_bx3 = torch.from_numpy(lightdirect)

    def forward(self, images):
        return self.model(images)

    def render_loss(self, pose_axis, images, targets):
        image_ref = images.permute(0, 2, 3, 1)
        nocs = targets['nocs']
        sil_ref = torch.zeros((nocs.shape[0], nocs.shape[2], nocs.shape[3]),
                              device=self.device)
        sil_ref[torch.sum(nocs, 1) != 0] = 1
        sil_ref = sil_ref[:, :, :, None]

        r_co = angle_axis_to_rotation_matrix(pose_axis[:, :3])[0]
        mat_co = torch.eye(4).to(self.device)
        rotation = r_co
        translation = pose_axis[0, 3:]
        mat_co[:3, :3] = rotation
        mat_co[:3, 3] = translation
        cam_rot, cam_trans = make_camera_mat_from_mat(mat_co, self.device)
        camera_params = [
            cam_rot[None].to(self.device), cam_trans[None].to(self.device),
            self.camera_proj_mat.to(self.device)
        ]

        # Visual Alignment
        # silhouette: [1, 480, 640, 1]
        # predictions: [1, 480, 640, 3]
        # torch.Size([1, 6078, 3]
        # torch.Size([12152, 3])
        # torch.Size([1, 3, 3])
        # 1,3
        # 3,1
        # 1, 6078, 2
        # [2048, 2048, 3]
        predictions, silhouette, _ = self.renderer(
            points=[
                self.vertices.to(self.device),
                self.faces.long().to(self.device)
            ],
            camera_params=camera_params,
            uv_bxpx2=self.uv.to(self.device),
            texture_bx3xthxtw=self.texture.to(self.device),
            lightdirect_bx3=self.tflight_bx3.to(self.device),
            material_bx3x3=self.tfmat.to(self.device),
            shininess_bx1=self.tfshi.to(self.device))
        loss = 0
        loss += torch.mean((predictions - image_ref)**2)
        loss += torch.mean((silhouette - sil_ref)**2)
        return loss, predictions, silhouette

    def training_step(self, batch, batch_idx):
        images, targets = batch
        pts3d = self.pts3d.to(self.device).type_as(images)
        K = self.K.to(self.device).type_as(images)
        self.ini_pose = self.ini_pose.to(self.device)

        pose_mat_gt = targets['pose']
        # pose_rot_gt = pose_mat_gt[:, 0, :3, :3]
        # rot_axis_gt = kornia.rotation_matrix_to_angle_axis(pose_rot_gt.clone())
        # pose_axis_gt = torch.cat([rot_axis_gt, pose_mat_gt[:, 0, :3, 3]],
        #                          dim=1)
        # pts2d_gt = BPnP.batch_project(pose_axis_gt, pts3d, K)

        pts2d_gt = BPnP.batch_project(pose_mat_gt[:, 0, :3],
                                      pts3d,
                                      K,
                                      angle_axis=False)

        pose_kpts2d = self.forward(images).reshape(1, self.n, 2)
        pose_axis = bpnp(pose_kpts2d, pts3d, K, self.ini_pose)

        # loss, _, _ = self.render_loss(pose_axis, images, targets)
        # self.log('train_loss', loss)
        # return loss

        pts2d_pro = BPnP.batch_project(pose_axis, pts3d, K)
        # TODO(taku): please add gt
        loss = ((pts2d_pro - pts2d_gt)**2).mean() + \
            ((pts2d_pro - pose_kpts2d)**2).mean()
        self.log('train_loss', loss)

        self.ini_pose = pose_axis.detach()
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        pts3d = self.pts3d.to(self.device).type_as(images)
        K = self.K.to(self.device).type_as(images)
        pose_mat_gt = targets['pose']

        # pose_rot_gt = pose_mat_gt[:, 0, :3, :3]
        # rot_axis_gt = kornia.rotation_matrix_to_angle_axis(pose_rot_gt.clone())
        # rot_axis_gt = kornia.quaternion_to_angle_axis(
        #     kornia.rotation_matrix_to_quaternion(pose_rot_gt.clone()))
        # pose_axis_gt = torch.cat([rot_axis_gt, pose_mat_gt[:, 0, :3, 3]],
        #                          dim=1)
        # pts2d_gt = BPnP.batch_project(pose_axis_gt, pts3d, K)
        pts2d_gt = BPnP.batch_project(pose_mat_gt[:, 0, :3],
                                      pts3d,
                                      K,
                                      angle_axis=False)

        pose_kpts2d = self.forward(images).reshape(1, self.n, 2)
        pose_axis = bpnp(pose_kpts2d, pts3d, K)
        pts2d_pro = BPnP.batch_project(pose_axis, pts3d, K)

        r_co = angle_axis_to_rotation_matrix(pose_axis[:, :3])[0]
        mat_co = torch.eye(4).to(self.device)
        rotation = r_co
        translation = pose_axis[0, 3:]
        mat_co[:3, :3] = rotation
        mat_co[:3, 3] = translation
        predicted_kpts = project(ten2num(pts3d), ten2num(K),
                                 ten2num(mat_co[:3]))

        image = draw_keypoints(images[0], ten2num(pose_kpts2d).astype(np.int))
        image = draw_keypoints(image,
                               ten2num(pts2d_gt).astype(np.int), (255, 0, 0))
        image = draw_keypoints(image, predicted_kpts[None].astype(np.int),
                               (0, 255, 0))
        self.logger.experiment.log({
            f"image{self.trainer.global_step}_{batch_idx}": [
                wandb.Image(image),
            ]
        })

        # TODO(taku): please add gt
        loss = ((pts2d_pro - pts2d_gt)**2).mean() + \
            ((pts2d_pro - pose_kpts2d)**2).mean()
        self.log('val_loss', loss)

    # def configure_optimizers(self):
    #     lr = 0.000004  # 0.0002
    #     lr_drop = 200
    #     weight_decay = 0.000001  # 0.0001
    #     optimizer = torch.optim.AdamW(self.parameters(),
    #                                   lr=lr,
    #                                   weight_decay=weight_decay)
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)
    #     return [optimizer], [lr_scheduler]

    def configure_optimizers(self):
        lr = 0.000004  # 0.0002
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return [optimizer], []


def run_model():
    seed_everything(777)

    if params['checkpoint'] == '':
        model = PoseModel()
    else:
        # checkpoint loading:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2550
        model = PoseModel()
        checkpoint = torch.load(params['checkpoint'])
        model.load_state_dict(checkpoint['state_dict'])

    log_name = str(datetime.datetime.now())
    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=f'lightning_logs/{log_name}/')
    log_name = str(datetime.datetime.now())
    wandb_logger = WandbLogger(name=log_name, project='lightning_logs')
    trainer = pl.Trainer(checkpoint_callback=checkpointer,
                         logger=wandb_logger,
                         gpus='0',
                         precision=32,
                         max_epochs=500)

    dm = VGGPoseDataModule()
    trainer.fit(model, dm)


if __name__ == "__main__":
    run_model()
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.vgg11()
        self.model.classifier = torch.nn.Linear(25088, n * 2)
        self.pts3d = torch.from_numpy(kpts3d)
        self.K = torch.from_numpy(K)

    def forward(self, image):
        images, targets = batch
        pts2d = model(image).reshape(1, n, 2)
        pose_axis = bpnp(pts2d, self.pts3d, self.K)
        pts2d_pro = BPnP.batch_project(pose_axis, pts3d_gt, K)
        # TODO(taku): please add gt
        loss = ((pts2d_pro - pts2d_gt)**2).mean() + \
            ((pts2d_pro - pts2d)**2).mean()


# TODO(taku): simplify more,
class Model(nn.Module):
    def __init__(self, renderer, image_ref, sil_ref, device, verticesc, facesc,
                 uvc, texturec, tflight_bx3c, tfmatc, tfshic, fx, fy, cx, cy,
                 pts2d_init_np, bpnp, K):
        super().__init__()
        self.renderer = renderer
        self.device = device

        # Get the reference silhouette and RGB image, points
        self.register_buffer('image_ref', image_ref)
        self.register_buffer('sil_ref', sil_ref)

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

        loss = 0
        loss += torch.mean((predictions - self.image_ref)**2)
        loss += torch.mean((silhouette - self.sil_ref)**2)
        return loss, predictions, silhouette, None


if __name__ == "__main__":
    n = 8
    device = 'cuda'
    model = torchvision.models.vgg11()
    model.classifier = torch.nn.Linear(25088, n * 2)
    model.to(device)

    pts2d = model(torch.ones(1, 3, 32, 32, device=device)).reshape(1, n, 2)
    pts3d = torch.from_numpy(kpts3d).to(device)
    K = torch.from_numpy(K).to(device)
    pose_axis = bpnp(pts2d, pts3d, K)

    # pose_rot = angle_axis_to_rotation_matrix(pose_axis[:, :3])[0]
    # pose_trans = pose_axis[:, 3:][0]
    '''
    mat_co = torch.eye(4).to(device)
    mat_co[:3, :3] = angle_axis_to_rotation_matrix(pose_axis[:, :3])[0]
    mat_co[:3, 3] = pose_axis[:, 3:][0]
    cam_rot, cam_trans = make_camera_mat_from_mat(mat_co, device)

    camera_proj_mat = torch.FloatTensor(camera_proj_mat_np).to(device)
    camera_params = [
        cam_rot[None],
        cam_trans[None], camera_proj_mat
    ]

    # vertices, facenp, uv
    pointnp_px3, facenp_fx3, uv = loadobjtex('obj_000001.obj')
    vertices = torch.from_numpy(pointnp_px3).to(device)
    vertices = vertices.unsqueeze(0)
    faces = torch.from_numpy(facenp_fx3).to(device)
    uv = torch.from_numpy(uv).type_as(vertices)
    uv = uv.unsqueeze(0)  # 1, 6078, 2

    # Texture
    texture = np.array(Image.open(params['text_path']))
    texture = torch.from_numpy(texture).to(device)
    texture = texture.float() / 255.0

    # Phong Setting
    bs = 1
    material = np.array(
        [[0.7, 0.7, 0.7], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]],
        dtype=np.float32).reshape(-1, 3, 3)
    # material = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0., 0., 0.]],
    #                     dtype=np.float32).reshape(-1, 3, 3)
    self.tfmat = torch.from_numpy(material).repeat(bs, 1, 1).to(device)
    shininess = np.array([100], dtype=np.float32).reshape(-1, 1)
    # shininess = np.array([0], dtype=np.float32).reshape(-1, 1)
    self.tfshi = torch.from_numpy(shininess).repeat(bs, 1).to(device)
    lightdirect = np.array([[1.0], [1.0], [0.5]]).astype(np.float32)
    # lightdirect = np.array([[0.0], [0.0], [0.0]]).astype(np.float32)
    self.tflight_bx3 = torch.from_numpy(lightdirect).to(device)

    predictions, silhouette, _ = self.renderer(
        points=[self.vertices, self.faces.long()],
        camera_params=camera_params,
        uv_bxpx2=self.uv,
        texture_bx3xthxtw=self.texture,
        lightdirect_bx3=self.tflight_bx3.to(device),
        material_bx3x3=self.tfmat.to(device),
        shininess_bx1=self.tfshi.to(device))

    import pdb
    pdb.set_trace()

    print(model)
    summary(model.cuda(), input_size=(3, 256, 256))
    # summary(model.cuda(), input_size=(3, 128, 128))
    # input_tensor = torch.zeros((1, 3, 256, 256))  # .cuda()
    # input_tensor = torch.zeros((1, 3, 128, 128))  # .cuda()
    # nocs, prob = model(input_tensor)
    # import pdb
    # pdb.set_trace()
    # print(model)
    '''
"""