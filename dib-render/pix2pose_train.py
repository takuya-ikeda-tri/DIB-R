import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from torch.nn import functional as F
import torch.nn as nn

params = {
    'input_size': 128,
    'checkpoint': '',
    'file_path': 'saved_models/',
}


class P2PDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()


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


class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x):
        return self.generator(x)

    def generator_loss(self, batch, alpha=3, beta=50):
        images, targets = batch
        gan_label = torch.ones(images.size(0), 1, self.device)
        nocs, prob = self.generator(batch)
        gan_result = self.discriminator(nocs)
        mask = torch.zeros((nocs.shape))
        mask[torch.nonzero(nocs)] = 1
        mask = mask[:, 0, :, :]
        loss_nocs = torch.sum(torch.abs(nocs - targets['nocs'], 1))
        loss_prob = F.mse_loss(prob[:, 0, :, :], torch.min(loss_nocs, 1))
        loss_nocs = torch.mean(loss_nocs * mask * alpha + loss_nocs *
                               (1 - mask)) * 100
        loss_gen = F.binary_cross_entropy(gan_result, gan_label)
        loss = loss_gen + beta * loss_prob + loss_nocs
        return loss

    def discriminator_loss(self, batch):
        images, targets = batch
        gan_label_real = torch.ones(images.size(0), 1, device=self.device)
        gan_label_fake = torch.zeros(images.size(0), 1, device=self.device)
        output_real = self.discriminator(images)
        loss_real = F.binary_cross_entropy(output_real, gan_label_real)
        output_fake = self.discriminator(self.generator(images))
        loss_fake = F.binary_cross_entropy(output_fake, gan_label_fake)
        loss = loss_real + loss_fake
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(batch)
        if optimizer_idx == 1:
            result = self.discriminator_step(batch)
        return result

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
    seed_everything(42)

    if params['checkpoint'] == '':
        model = GAN()
    else:
        # checkpoint loading:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2550
        model = GAN()
        checkpoint = torch.load(params['checkpoint'])
        model.load_state_dict(checkpoint['state_dict'])

    checkpointer = pl.callbacks.ModelCheckpoint(filepath=params['file_path'])
    trainer = pl.Trainer(checkpoint_callback=checkpointer,
                         gpus=1,
                         precision=16,
                         max_epochs=100)
    # dm = P2PDataModule()
    # trainer.fit(model, dm)


if __name__ == "__main__":
    run_model()
