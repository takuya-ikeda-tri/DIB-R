import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

import datetime
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger

from pix2pose_dataset import P2PDataset
from pix2pose_dataset import build_transform
from pix2pose_dataset import loadobjtex
from utils.image_utils import solve_pnp
from utils.object_pose_estimation_center_utils import draw_keypoints

params = {
    'obj_path': './obj_000001.obj',
    'height': 128,
    'width': 128,
    'input_size': 128,
    'checkpoint': "./lightning_logs/epoch=80.ckpt",
    'file_path': 'test',
    'batch_size': 4,
    'num_workers': 8,
    'split_ratio': 0.05,
    'vert_max': 0.1048,
    'vert_min': -0.1048
}

KPT_3D = np.array([[-0.0334153, -0.0334686, -0.104798],
                   [-0.0334153, -0.0334686, 0.104798],
                   [-0.0334153, 0.0334686, -0.104798],
                   [-0.0334153, 0.0334686, 0.104798],
                   [0.0334153, -0.0334686, -0.104798],
                   [0.0334153, -0.0334686, 0.104798],
                   [0.0334153, 0.0334686, -0.104798],
                   [0.0334153, 0.0334686, 0.104798]])

# KPT_3D = np.array([[-0.0334153, -0.0334686, -0.104798],
#                    [-0.0334153, -0.0334686, 0.104798],
#                    [-0.0334153, 0.0334686, -0.104798],
#                    [-0.0334153, 0.0334686, 0.104798],
#                    [0.0334153, -0.0334686, -0.104798],
#                    [0.0334153, -0.0334686, 0.104798],
#                    [0.0334153, 0.0334686, -0.104798],
#                    [0.0334153, 0.0334686, 0.104798], [0., 0., 0.]])


def denorm():
    pointnp_px3, facenp_fx3, uv = loadobjtex(params['obj_path'])
    vertices = torch.from_numpy(pointnp_px3)
    vertices = vertices.unsqueeze(0)
    vert_min = torch.min(vertices)
    vert_max = torch.max(vertices)
    # colors = (self.vertices - vert_min) / (vert_max - vert_min)
    return vert_max, vert_min  # (tensor(0.1048), tensor(-0.1048))


def project(points_3d, intrinsics, pose):
    points_3d = np.dot(points_3d, pose[:, :3].T) + pose[:, 3:].T
    points_3d = np.dot(points_3d, intrinsics.T)
    points_2d = points_3d[:, :2] / points_3d[:, 2:]
    return points_2d


class P2PDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = P2PDataset

    def setup(self, stage):
        train_transform = build_transform(True)
        val_transform = build_transform(False)
        if stage == 'fit':
            train_data = self.dataset(train_transform)
            val_data = self.dataset(val_transform)
            indices = torch.randperm(len(train_data)).tolist()
            split_th = int(len(train_data) * params['split_ratio'])
            self.train_data = torch.utils.data.Subset(train_data,
                                                      indices[:-split_th])
            self.val_data = torch.utils.data.Subset(val_data,
                                                    indices[-split_th:])
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


# TODO(taku): refactor this model following
# https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/gans/basic/components.py
class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, d, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(d)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(d, 2 * d, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(2 * d)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv2d(2 * d, 4 * d, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(4 * d)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.conv4 = nn.Conv2d(4 * d, 8 * d, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(8 * d)
        self.relu4 = nn.LeakyReLU(inplace=True)

        self.conv5 = nn.Conv2d(8 * d, 8 * d, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(8 * d)
        self.relu5 = nn.LeakyReLU(inplace=True)

        self.conv6 = nn.Conv2d(8 * d, 8 * d, 3, 2, 1)
        self.bn6 = nn.BatchNorm2d(8 * d)
        self.relu6 = nn.LeakyReLU(inplace=True)

        self.fc1 = nn.Linear(
            int(params['input_size'] / 2**6) *
            int(params['input_size'] / 2**6) * d * 8, 1)

    def forward(self, x):
        f1 = self.relu1(self.bn1(self.conv1(x)))  # 64x64x64
        f2 = self.relu2(self.bn2(self.conv2(f1)))  # 128x32x32

        f3 = self.relu3(self.bn3(self.conv3(f2)))  # 256x16x16
        f4 = self.relu4(self.bn4(self.conv4(f3)))  # 512x8x8
        f5 = self.relu5(self.bn5(self.conv5(f4)))  # 512x4x4
        f6 = self.relu6(self.bn6(self.conv6(f5)))  # 512x2x2
        x = f6.view(f6.size(0), -1)
        x = self.fc1(x)
        return torch.sigmoid(x)


# TODO(taku): should do weights init?
class Generator(nn.Module):
    def __init__(self, d=64):
        super(Generator, self).__init__()
        # f1
        # input channel, output channel, kernel_size, stride, padding
        # (H + 2 x padding - dilation x (kernel_size - 1) - 1)/stride + 1
        self.conv1 = nn.Conv2d(3, d, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(d)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, d, 5, 2, 2)
        self.bn2 = nn.BatchNorm2d(d)
        self.relu2 = nn.LeakyReLU(inplace=True)

        # f2
        self.conv3 = nn.Conv2d(d * 2, d * 2, 5, 2, 2)
        self.bn3 = nn.BatchNorm2d(d * 2)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.conv4 = nn.Conv2d(d * 2, d * 2, 5, 2, 2)
        self.bn4 = nn.BatchNorm2d(d * 2)
        self.relu4 = nn.LeakyReLU(inplace=True)

        # f3
        self.conv5 = nn.Conv2d(d * 4, d * 2, 5, 2, 2)
        self.bn5 = nn.BatchNorm2d(d * 2)
        self.relu5 = nn.LeakyReLU(inplace=True)

        self.conv6 = nn.Conv2d(d * 4, d * 2, 5, 2, 2)
        self.bn6 = nn.BatchNorm2d(d * 2)
        self.relu6 = nn.LeakyReLU(inplace=True)

        # f4
        self.conv7 = nn.Conv2d(d * 4, d * 4, 5, 2, 2)
        self.bn7 = nn.BatchNorm2d(d * 4)
        self.relu7 = nn.LeakyReLU(inplace=True)

        self.conv8 = nn.Conv2d(d * 4, d * 4, 5, 2, 2)
        self.bn8 = nn.BatchNorm2d(d * 4)
        self.relu8 = nn.LeakyReLU(inplace=True)

        # encode
        self.fc1 = nn.Linear(
            int(params['input_size'] / 16) * int(params['input_size'] / 16) *
            d * 8, d * 4)
        self.fc2 = nn.Linear(
            d * 4,
            int(params['input_size'] / 16) * int(params['input_size'] / 16) *
            d * 4)

        # d1
        # input_channel, output_channel, kernel_size, stride, padding
        # (H - 1) x stride - 2 x padding + dilation x (kernel_size-1) + \
        # output_padding + 1
        self.conv9 = nn.ConvTranspose2d(d * 4, d * 2, 5, 2, 2, 1)
        self.bn9 = nn.BatchNorm2d(d * 2)
        self.relu9 = nn.LeakyReLU(inplace=True)

        # d1_uni
        self.conv10 = nn.ConvTranspose2d(d * 4, d * 4, 5, 1, 2)
        self.bn10 = nn.BatchNorm2d(d * 4)
        self.relu10 = nn.LeakyReLU(inplace=True)

        # d2
        self.conv11 = nn.ConvTranspose2d(d * 4, d * 2, 5, 2, 2, 1)
        self.bn11 = nn.BatchNorm2d(d * 2)
        self.relu11 = nn.LeakyReLU(inplace=True)

        # d2_uni
        self.conv12 = nn.Conv2d(d * 4, d * 4, 5, 1, 2)
        self.bn12 = nn.BatchNorm2d(d * 4)
        self.relu12 = nn.LeakyReLU(inplace=True)

        # d3
        self.conv13 = nn.ConvTranspose2d(d * 4, d, 5, 2, 2, 1)
        self.bn13 = nn.BatchNorm2d(d)
        self.relu13 = nn.LeakyReLU(inplace=True)

        # d3_uni
        self.conv14 = nn.Conv2d(d * 2, d * 2, 5, 1, 2)
        self.bn14 = nn.BatchNorm2d(d * 2)
        self.relu14 = nn.LeakyReLU(inplace=True)

        # decode
        self.conv15 = nn.ConvTranspose2d(d * 2, 1, 5, 2, 2, 1)
        self.conv16 = nn.ConvTranspose2d(d * 2, 3, 5, 2, 2, 1)

    def forward(self, x):
        f1_1 = self.relu1(self.bn1(self.conv1(x)))  # 64x64x64
        f1_2 = self.relu2(self.bn2(self.conv2(x)))  # 64x64x64

        f1 = torch.cat((f1_1, f1_2), 1)  # 128x64x64

        f2_1 = self.relu3(self.bn3(self.conv3(f1)))  # 128x32x32
        f2_2 = self.relu4(self.bn4(self.conv4(f1)))  # 128x32x32

        f2 = torch.cat((f2_1, f2_2), 1)  # 256x32x32

        f3_1 = self.relu5(self.bn5(self.conv5(f2)))  # 128x16x16
        f3_2 = self.relu6(self.bn6(self.conv6(f2)))  # 128x16x16

        f3 = torch.cat((f3_1, f3_2), 1)  # 256x16x16

        f4_1 = self.relu7(self.bn7(self.conv7(f3)))  # 256x8x8
        f4_2 = self.relu8(self.bn8(self.conv8(f3)))  # 256x8x8

        f4 = torch.cat((f4_1, f4_2), 1)  # 512x8x8

        x = f4.view(f4.size(0), -1)
        encode = self.fc1(x)  # 256

        d1 = self.fc2(encode)  # 256x8x8
        d1 = self.relu9(
            self.bn9(
                self.conv9(
                    d1.view(-1, 256, int(params['input_size'] / 16),
                            int(params['input_size'] / 16)))))  # 128x16x16

        d1_uni = torch.cat((d1, f3_2), 1)  # 256x16x16
        d1_uni = self.relu10(self.bn10(self.conv10(d1_uni)))  # 256x16x16

        d2 = self.relu11(self.bn11(self.conv11(d1_uni)))  # 128x32x32

        d2_uni = torch.cat((d2, f2_2), 1)  # 256x32x32
        d2_uni = self.relu12(self.bn12(self.conv12(d2_uni)))  # 256x32x32

        d3 = self.relu13(self.bn13(self.conv13(d2_uni)))  # 64x64x64

        d3_uni = torch.cat((d3, f1_2), 1)  # 128x64x64
        d3_uni = self.relu14(self.bn14(self.conv14(d3_uni)))  # 128x64x64

        prob = self.conv15(d3_uni)  # 1x128x128
        nocs = self.conv16(d3_uni)  # 1x128x128

        return torch.tanh(nocs), torch.sigmoid(prob)


def ten2num(input_tensor, ttype=torch.FloatTensor):
    return input_tensor.type(ttype).numpy()


class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.K = np.eye(3).astype(np.float32)
        self.K[0, 0] = 200.0
        self.K[1, 1] = 200.0
        self.K[0, 2] = params['width'] / 2.0
        self.K[1, 2] = params['height'] / 2.0

    def forward(self, x):
        return self.generator(x)

    def generator_loss(self, batch, alpha=3, beta=50):
        images, targets = batch
        gan_label = torch.ones((images.size(0), 1), device=self.device)
        nocs, prob = self.generator(images)
        gan_result = self.discriminator(nocs)
        mask = torch.zeros((nocs.shape[0], nocs.shape[2], nocs.shape[3]),
                           device=self.device)
        # mask[torch.sum(nocs, 1) != 0] = 1
        mask[torch.sum(targets['nocs'], 1) != 0] = 1

        loss_nocs = torch.sum(torch.abs(nocs - targets['nocs']), 1)
        # TODO(taku): have to divide the loss_nocs // 3.0??
        # loss_prob = F.mse_loss(prob[:, 0, :, :], torch.min(loss_nocs, 1))
        min_loss_nocs = loss_nocs.clone()
        min_loss_nocs[loss_nocs > 1] = 1.0
        loss_prob = F.mse_loss(prob[:, 0, :, :], min_loss_nocs)
        loss_nocs = torch.mean(loss_nocs * mask * alpha + loss_nocs *
                               (1 - mask)) * 100
        # loss_gen = F.binary_cross_entropy(gan_result,
        #                                   gan_label.type_as(gan_result))
        loss_gen = F.binary_cross_entropy_with_logits(gan_result, gan_label)
        loss = loss_gen + beta * loss_prob + loss_nocs
        return loss

    def discriminator_loss(self, batch):
        images, targets = batch
        gan_label_real = torch.ones((images.size(0), 1), device=self.device)
        gan_label_fake = torch.zeros((images.size(0), 1), device=self.device)
        output_real = self.discriminator(images)
        # loss_real = F.binary_cross_entropy(output_real, gan_label_real)
        loss_real = F.binary_cross_entropy_with_logits(output_real,
                                                       gan_label_real)
        output_fake = self.discriminator(self.generator(images)[0])
        # loss_fake = F.binary_cross_entropy(output_fake, gan_label_fake)
        loss_fake = F.binary_cross_entropy_with_logits(output_fake,
                                                       gan_label_fake)
        loss = loss_real + loss_fake
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(batch)
        if optimizer_idx == 1:
            result = self.discriminator_step(batch)
        return result

    def get_kpt(self, prob, nocs):
        pass

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # First iteration
        nocs, prob = self.generator(images)
        mask = torch.zeros(nocs[:, 0, :, :].shape).type_as(prob)
        mask_th = 0.5
        mask[torch.sum(nocs, dim=1) > mask_th] = 1
        prob = prob * mask[None]
        prob_mask = torch.zeros(nocs[:, 0, :, :].shape).type_as(prob)
        # prob_th = 0.1
        prob_th = 0.1
        prob_mask[(0 < prob[:, 0]) & (prob_th > prob[:, 0])] = 1
        # nocs = nocs * mask[None] * prob_mask[None]
        nocs = nocs * prob_mask[None]

        # Second itreation
        # images = images * prob_mask[None]
        # nocs, prob = self.generator(images)
        # mask = torch.zeros(nocs[:, 0, :, :].shape).type_as(prob)
        # mask_th = 0.5
        # mask[torch.sum(nocs, dim=1) > mask_th] = 1
        # prob = prob * mask[None]
        # prob_mask = torch.zeros(nocs[:, 0, :, :].shape).type_as(prob)
        # # prob_th = 0.1
        # prob_th = 0.1
        # prob_mask[(0 < prob[:, 0]) & (prob_th > prob[:, 0])] = 1
        # # nocs = nocs * mask[None] * prob_mask[None]
        # nocs = nocs * prob_mask[None]

        kpt2d = nocs[0, 0].nonzero()
        kpt3d = nocs[0, :, kpt2d[:, 0], kpt2d[:, 1]].permute(1, 0)
        # colors = (self.vertices - vert_min) / (vert_max - vert_min)
        kpt3d = kpt3d * (params['vert_max'] - params['vert_min']) + \
            params['vert_min']
        kpt2d = kpt2d[:, [1, 0]]

        xmax = torch.max(kpt3d[:, 0])
        xmin = torch.min(kpt3d[:, 0])
        ymax = torch.max(kpt3d[:, 1])
        ymin = torch.min(kpt3d[:, 1])
        zmax = torch.max(kpt3d[:, 2])
        zmin = torch.min(kpt3d[:, 2])

        xmax_coord = kpt2d[torch.argmax(kpt3d[:, 0])]
        xmin_coord = kpt2d[torch.argmin(kpt3d[:, 0])]
        ymax_coord = kpt2d[torch.argmax(kpt3d[:, 1])]
        ymin_coord = kpt2d[torch.argmin(kpt3d[:, 1])]
        zmax_coord = kpt2d[torch.argmax(kpt3d[:, 2])]
        zmin_coord = kpt2d[torch.argmin(kpt3d[:, 2])]
        coord_list = [
            xmax_coord[None], xmin_coord[None], ymax_coord[None],
            ymin_coord[None], zmax_coord[None], zmin_coord[None]
        ]
        coord_2d = torch.cat(coord_list, dim=0)

        xmax_xyz = kpt3d[torch.argmax(kpt3d[:, 0])]
        xmin_xyz = kpt3d[torch.argmin(kpt3d[:, 0])]
        ymax_xyz = kpt3d[torch.argmax(kpt3d[:, 1])]
        ymin_xyz = kpt3d[torch.argmin(kpt3d[:, 1])]
        zmax_xyz = kpt3d[torch.argmax(kpt3d[:, 2])]
        zmin_xyz = kpt3d[torch.argmin(kpt3d[:, 2])]
        xyz_list = [
            xmax_xyz[None], xmin_xyz[None], ymax_xyz[None], ymin_xyz[None],
            zmax_xyz[None], zmin_xyz[None]
        ]
        xyz_3d = torch.cat(xyz_list)
        result1 = solve_pnp(ten2num(xyz_3d), ten2num(coord_2d), self.K)
        result2 = solve_pnp(ten2num(kpt3d), ten2num(kpt2d), self.K)
        projected_kpt = project(KPT_3D, self.K, result2)
        image = draw_keypoints(images[0], projected_kpt[None].astype(np.int))

        # nocs[:,0] -> x axis, nocs[:,1] -> y axis, nocs[:,2] -> z axis

        # result = solve_pnp(
        #     ten2num(kpt3d)[:, [2, 1, 0]], ten2num(kpt2d), self.K)
        # result = solve_pnp(ten2num(kpt3d), ten2num(kpt2d), self.K)
        # result = solve_pnp(ten2num(kpt3d), ten2num(kpt2d)[:, [1, 0]], self.K)
        # result = solve_pnp(
        #     ten2num(kpt3d)[:, [2, 1, 0]],
        #     ten2num(kpt2d)[:, [1, 0]], self.K)

        # nocs = nocs * mask[None]
        # topk = torch.topk(prob[prob != 0], k=100, dim=0, largest=False)
        # import pdb
        # pdb.set_trace()

        # ten2num(prob)

        # idx = torch.topk(prob[prob != 0], k=100, dim=0)[1]
        # prob[prob != 0].scatter_(0, )

        # idx = torch.topk(prob[0], k=2)
        # prob[0].scatter_(0, idx, prob.shape[2] * prob.shape[3])

        if batch_idx in [0, 1, 2, 3, 4, 5]:
            self.logger.experiment.log({
                f"image{self.trainer.global_step}_{batch_idx}": [
                    # wandb.Image(images[0]),
                    wandb.Image(image),
                    wandb.Image(targets['nocs'][0]),
                    wandb.Image(mask),
                    wandb.Image(nocs[0]),
                    wandb.Image(prob[0])
                ]
            })
            # self.logger.experiment.log({
            #     f"image{self.trainer.global_step}_{batch_idx}": [
            #         wandb.Image(images[0]),
            #         wandb.Image(targets['nocs'][0]),
            #         wandb.Image(nocs[0, 0]),
            #         wandb.Image(nocs[0, 1]),
            #         wandb.Image(nocs[0, 2])
            #     ]
            # })
            print("max-min", xmax, xmin, ymax, ymin, zmax, zmin)
            print("max-min-coord", xmax_coord, xmin_coord, ymax_coord,
                  ymin_coord, zmax_coord, zmin_coord)
            # import pdb
            # pdb.set_trace()

    def generator_step(self, batch):
        g_loss = self.generator_loss(batch)
        self.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_step(self, batch):
        d_loss = self.discriminator_loss(batch)
        self.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
        return d_loss

    def configure_optimizers(self):
        lr = 0.0002
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []


def run_model():
    seed_everything(777)

    if params['checkpoint'] == '':
        model = GAN()
    else:
        # checkpoint loading:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2550
        model = GAN()
        checkpoint = torch.load(params['checkpoint'])
        model.load_state_dict(checkpoint['state_dict'])

    checkpointer = pl.callbacks.ModelCheckpoint(dirpath='lightning_logs')
    # checkpointer = pl.callbacks.ModelCheckpoint(filepath=params['file_path'])
    # trainer = pl.Trainer(checkpoint_callback=checkpointer,
    #                      gpus='0',
    #                      precision=16,
    #                      max_epochs=100)

    log_name = str(datetime.datetime.now())
    wandb_logger = WandbLogger(name=log_name, project='lightning_logs')
    trainer = pl.Trainer(checkpoint_callback=checkpointer,
                         logger=wandb_logger,
                         gpus='0',
                         precision=16,
                         max_epochs=100)

    dm = P2PDataModule()
    trainer.fit(model, dm)


if __name__ == "__main__":
    run_model()
