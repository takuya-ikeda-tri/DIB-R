"""Training and evaluation for 6d pose estimation model."""
# config and logger depends
import hydra
from omegaconf import DictConfig
import wandb

import random
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchvision import models
from torch.utils.data import DataLoader
import kornia
from PIL import Image

from losses import object_pose_estimation_center_losses
from metrics import object_pose_estimation_center_metrics
from datasets import object_pose_estimation_center_dataset
from utils.object_pose_estimation_center_utils import extract_peaks_from_centroid
from utils.object_pose_estimation_center_utils import extract_keypoints_peakvoting
from utils.object_pose_estimation_center_utils import extract_keypoints_peaks
from utils.object_pose_estimation_center_utils import draw_keypoints
from utils.object_pose_estimation_center_utils import draw_text

import BPnP
from simple_renderer import Renderer


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


def make_camera_mat_from_mat(mat):
    mat[0, 3] = -1 * mat[0, 3]
    conv_mat3 = torch.eye(3).type_as(mat)
    conv_mat3[1, 1] = -1.0
    conv_mat3[2, 2] = -1.0
    camera_r_param = conv_mat3 @ mat[:3, :3]
    tes_conv_matrix2 = torch.eye(4).type_as(mat)
    # tes_conv_matrix2[:3, :3] = torch.inverse(camera_r_param)
    tes_conv_matrix2[:3, :3] = camera_r_param.t()
    camera_t_param = (tes_conv_matrix2 @ mat)[:3, 3]
    # test_conv_matrix2 is Roc? camera_t_param is Toc
    return camera_r_param, camera_t_param


def angle_axis_to_rotation_matrix(angle_axis):
    rotation_matrix = kornia.quaternion_to_rotation_matrix(
        kornia.angle_axis_to_quaternion(angle_axis))
    return rotation_matrix


def get_bbox_vertices_from_vertex(vertex_fields, index, scale_factor=1):
    """Get 8 vertices of bouding box from vertex displacement fields.
    Args:
        vertex_fields (numpy.array): (height, width, 16)
        index (numpy.array): (2)
        scale_factor (int, optional): Defaults to 1.
    Returns:
        [type]: (8,2)
    """
    assert index.shape[0] == 2
    index[0] = int(index[0] / scale_factor)
    index[1] = int(index[1] / scale_factor)
    vertices = vertex_fields[index[0], index[1], :]
    vertices = vertices.reshape([8, 2])
    vertices = scale_factor * index - vertices
    return vertices


def get_bbox_vertices_from_vertex_torch(vertex_fields, index, scale_factor=1):
    """Get 8 vertices of bouding box from vertex displacement fields.
    Args:
        vertex_fields (torch): (height, width, 16)
        index (torch): (2)
        scale_factor (int, optional): Defaults to 1.
    Returns:
        [type]: (8,2)
    """
    assert index.shape[0] == 2
    index[0] = (index[0] // scale_factor).int()
    index[1] = (index[1] // scale_factor).int()
    vertices = vertex_fields[index[0], index[1], :]
    vertices = vertices.reshape([8, 2])
    vertices = scale_factor * index - vertices
    return vertices


def extract_vertices_from_coords(coords, vertex_fields, img, scale_factor=1):
    """Extract keypoints from peaks and vertex displacement field.
    Args:
        peaks (numpy.array): (peak_num, 2)
        vertex_fields (numpy.array): (height, width, 16)
        img (numpy.array): (height, width, 16)
        scale_factor (int, optional): Defaults to 1.
    Returns:
        (numpy.array): (peak_num, 8, 2)
    """
    assert coords.shape[1] == 2
    assert vertex_fields.shape[2] == 16
    height, width = img.shape[0:2]
    # denormalize using height and width
    vertex_fields[:, :, ::2] = (1.0 - vertex_fields[:, :, ::2]) * (
        2 * height) - height
    vertex_fields[:, :,
                  1::2] = (1.0 - vertex_fields[:, :, 1::2]) * (2 *
                                                               width) - width
    vertices = np.zeros(
        (len(coords), vertex_fields.shape[2] // 2, coords.shape[1]))
    for i, coord in enumerate(coords):
        vertices[i] = get_bbox_vertices_from_vertex(vertex_fields,
                                                    coord,
                                                    scale_factor=scale_factor)
    return vertices


def extract_vertices_from_coords_torch(coords,
                                       vertex_fields,
                                       img,
                                       scale_factor=1):
    """Extract keypoints from peaks and vertex displacement field.
    Args:
        peaks (torch): (peak_num, 2)
        vertex_fields (torch): (height, width, 16)
        img (torch): (height, width, 16)
        scale_factor (int, optional): Defaults to 1.
    Returns:
        (torch): (peak_num, 8, 2)
    """
    assert coords.shape[1] == 2
    assert vertex_fields.shape[2] == 16
    height, width = img.shape[0:2]
    # denormalize using height and width
    vertex_fields[:, :, ::2] = (1.0 - vertex_fields[:, :, ::2]) * (
        2 * height) - height
    vertex_fields[:, :,
                  1::2] = (1.0 - vertex_fields[:, :, 1::2]) * (2 *
                                                               width) - width
    vertices = [
        get_bbox_vertices_from_vertex_torch(vertex_fields,
                                            coord,
                                            scale_factor=scale_factor)
        for coord in coords
    ]
    vertices = torch.cat(vertices).reshape(len(coords),
                                           vertex_fields.shape[2] // 2,
                                           coords.shape[1])
    return vertices


def extract_keypoints_from_coords(coords, vertex):
    """Extract keypoints from peaks and vertex displacement field.
    Args:
        coords(numpy.array): (xy_num, 2)
        vertex(numpy.array): (16, height, width)
    """
    # TODO(taku): refactor the below function to more simple
    kpts_2d = extract_vertices_from_coords(coords, vertex.transpose(1, 2, 0),
                                           vertex.transpose(1, 2, 0), 1)
    # Adjust the center point using peak value
    kpts_2d = kpts_2d - \
        (np.sum(np.array(kpts_2d), axis=1) / 8 - coords)[:, None]
    kpts_2d_with_center = np.concatenate([kpts_2d, coords[:, None, :]], 1)
    return kpts_2d_with_center[:, :, [1, 0]].astype(int)  # numpy array


def extract_keypoints_from_coords_torch(coords, vertex):
    """Extract keypoints from peaks and vertex displacement field.
    Args:
        coords(torch): (xy_num, 2)
        vertex(torch): (16, height, width)
    """
    # TODO(taku): refactor the below function to more simple
    kpts_2d = extract_vertices_from_coords_torch(coords,
                                                 vertex.permute(1, 2, 0),
                                                 vertex.permute(1, 2, 0), 1)
    # Adjust the center point using peak value
    kpts_2d = kpts_2d - \
        (torch.sum(kpts_2d, axis=1) / 8 - coords)[:, None]
    kpts_2d_with_center = torch.cat([kpts_2d, coords[:, None, :]], 1)
    return kpts_2d_with_center[:, :, [1, 0]]


def ten2num(input_tensor, ttype=torch.FloatTensor):
    return input_tensor.type(ttype).numpy()


# TODO(taku): consider the type hint, and shape for all script
# TODO(taku): consider the numpy or tensor for several calculation
class ObjectPoseEstimationModel(pl.LightningModule):
    """Pytorch Lightning module for training object_pose_estimation."""
    def __init__(self, params: DictConfig = None):
        super(ObjectPoseEstimationModel, self).__init__()
        """Initialize lightning module for object_pose_estimation training."""
        self.params = params

        self.backbone = models.segmentation.fcn_resnet50().backbone
        # 18 means 8bbox vertex + center x,y points
        self.segmentation_head = models.segmentation.fcn_resnet50(
            num_classes=18 + params.general.class_num).classifier
        self.heatmap_head = models.segmentation.fcn_resnet50(
            num_classes=1).classifier
        # 16 means 8bbox vertex
        self.vertex_head = models.segmentation.fcn_resnet50(
            num_classes=16).classifier

        self.metrics = \
            object_pose_estimation_center_metrics.ObjectPoseEstimationMetrics()
        self.loss = \
            object_pose_estimation_center_losses.ObjectPoseEstimationLoss()
        self.dataset = \
            object_pose_estimation_center_dataset.ObjectPoseEstimationDataset

        self.bpnp = BPnP.BPnP.apply
        renderer_mode = 'Phong'
        HEIGHT = 480
        WIDTH = 640
        self.renderer = Renderer(HEIGHT, WIDTH, mode=renderer_mode)

        objmesh_path = '/home/takuya/Projects/machine_learning/sandbox/DIB-R/dib-render/obj_000001.obj'
        objtexture_path = '/home/takuya/Projects/machine_learning/sandbox/DIB-R/dib-render/real_tea_bottle.png'
        pointnp_px3, facenp_fx3, uv = loadobjtex(objmesh_path)
        vertices = torch.from_numpy(pointnp_px3)
        self.vertices = vertices.unsqueeze(0)
        self.faces = torch.from_numpy(facenp_fx3)

        uv = torch.from_numpy(uv)
        self.uv = uv.unsqueeze(0)  # 1, 6078, 2
        texture = np.array(Image.open(objtexture_path))
        texture = torch.from_numpy(texture)
        texture = texture.float() / 255.0
        # Convert to NxCxHxW layout
        self.texture = texture.permute(2, 0, 1).unsqueeze(0)

        bs = len(vertices)
        material = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0., 0., 0.]],
                            dtype=np.float32).reshape(-1, 3, 3)
        self.tfmat = torch.from_numpy(material).repeat(bs, 1, 1)
        shininess = np.array([0], dtype=np.float32).reshape(-1, 1)
        self.tfshi = torch.from_numpy(shininess).repeat(bs, 1)
        lightdirect = np.array([[0.0], [0.0], [0.0]]).astype(np.float32)
        tflight = torch.from_numpy(lightdirect)
        self.tflight_bx3 = tflight

    # the training var length is len(train_data)//batch_size + len(val_data)
    def prepare_data(self):
        ann_path = hydra.utils.to_absolute_path(self.params.general.ann_path)
        base_path = hydra.utils.get_original_cwd()
        transform = object_pose_estimation_center_dataset.build_transform(
            is_training=True)
        train_data = self.dataset(ann_path, transform, base_path,
                                  self.params.general.max_instance_num)
        # print(train_data[0])
        transform = object_pose_estimation_center_dataset.build_transform(
            is_training=False)
        val_data = self.dataset(ann_path, transform, base_path,
                                self.params.general.max_instance_num)
        indices = torch.randperm(len(train_data)).tolist()
        split_th = int(len(train_data) * self.params.data.split_ratio)
        self.train_data = torch.utils.data.Subset(train_data,
                                                  indices[:-split_th])
        self.val_data = torch.utils.data.Subset(val_data, indices[-split_th:])

    def train_dataloader(self):
        # TODO(taku): consider batch issue which contains different shape data
        # TODO(taku): consider adding custom_collate
        # https://nodaki.hatenablog.com/entry/2018/07/18/001851
        # https://github.com/pytorch/pytorch/issues/1512
        # https://discuss.pytorch.org/t/dataloader-gives-stack-expects-each-tensor-to-be-equal-size-due-to-different-image-has-different-objects-number/91941/3
        train_data = DataLoader(self.train_data,
                                batch_size=self.params.data.batch_size,
                                shuffle=True,
                                num_workers=self.params.data.num_workers,
                                drop_last=True)
        return train_data

    def val_dataloader(self):
        val_data = DataLoader(self.val_data,
                              batch_size=1,
                              shuffle=False,
                              num_workers=self.params.data.num_workers,
                              drop_last=True)
        return val_data

    def test_dataloader(self):
        return self.val_dataloader()

    def forward(self, images):
        """Forward pass of the model."""
        input_shape = images.shape[-2:]
        outputs = self.backbone(images)['out']
        segmentation_outputs = self.segmentation_head(outputs)
        segmentation_outputs = F.interpolate(segmentation_outputs,
                                             size=input_shape,
                                             mode='bilinear',
                                             align_corners=False)
        heatmap_outputs = self.heatmap_head(outputs)
        heatmap_outputs = F.interpolate(heatmap_outputs,
                                        size=input_shape,
                                        mode='bilinear',
                                        align_corners=False)
        vertex_outputs = self.vertex_head(outputs)
        vertex_outputs = F.interpolate(vertex_outputs,
                                       size=input_shape,
                                       mode='bilinear',
                                       align_corners=False)
        return segmentation_outputs, heatmap_outputs, vertex_outputs

    def calculate_loss(self,
                       segmentations,
                       heatmaps,
                       vertexes,
                       targets,
                       idi=0):
        """Compute loss from predictions and ground truth."""
        individual_losses = {}
        mask_loss, heatmap_loss, vertex_loss, vote_loss = \
            self.loss(segmentations, heatmaps, vertexes, targets, self.params)

        individual_losses['mask_loss'] = mask_loss
        individual_losses['heatmap_loss'] = heatmap_loss
        # TODO(taku): modify these name for center_vertex and pixel_vertex
        individual_losses['vertex_loss'] = vertex_loss
        individual_losses['vote_loss'] = vote_loss

        loss = mask_loss + heatmap_loss + vertex_loss + vote_loss
        return loss, individual_losses

    def training_step(self, batch, batch_idx):
        """Do the work necessary for a single train step."""
        images, targets = batch
        segmentation_outputs, heatmap_outputs, vertex_outputs = \
            self.forward(images)

        loss = 0
        for idx in range(len(images)):
            mask = torch.argmax(
                segmentation_outputs[:, :self.params.general.class_num], dim=1)
            mask = mask[idx]
            heatmap_th = 0.9
            heatmap_data = (heatmap_outputs[idx, 0] > heatmap_th) * mask
            heatmap_nonzero_coord = torch.nonzero(heatmap_data)
            coords_kpts_2d = extract_keypoints_from_coords_torch(
                heatmap_nonzero_coord, vertex_outputs[idx])

            K = targets['K'][0]  # (3,3)
            pts3d_gt = targets['kpts_3d'][0][0]  # (9,3)

            camera_proj_mat_np = np.array(
                [[K[0, 0] / K[0, 2]], [K[1, 1] / K[1, 2]], [-1]],
                dtype=np.float32)
            camera_proj_mat = torch.from_numpy(camera_proj_mat_np).type_as(K)
            P_out = self.bpnp(coords_kpts_2d, pts3d_gt, K)
            Rco = angle_axis_to_rotation_matrix(P_out[:, :3])
            camera_params = []
            for po, rco in zip(P_out, Rco):
                mat_co = torch.eye(4).type_as(images)
                mat_co[:3, :3] = rco
                mat_co[:3, 3] = po[3:]
                cam_rot, cam_trans = make_camera_mat_from_mat(mat_co)
                camera_r_param = cam_rot.type_as(images)
                camera_t_param = cam_trans.type_as(images)
                # TODO(taku): consider the batch way
                camera_params = [
                    camera_r_param[None], camera_t_param[None],
                    camera_proj_mat.type_as(images)
                ]
                predictions, silhouette, _ = self.renderer(
                    points=[
                        self.vertices.type_as(images),
                        self.faces.type_as(images).long()
                    ],
                    camera_params=camera_params,
                    uv_bxpx2=self.uv.type_as(images),
                    texture_bx3xthxtw=self.texture.type_as(images),
                    lightdirect_bx3=self.tflight_bx3.type_as(images),
                    material_bx3x3=self.tfmat.type_as(images),
                    shininess_bx1=self.tfshi.type_as(images))
                import pdb
                pdb.set_trace()

                loss += torch.mean((predictions - images[idx])**2)
                loss += torch.mean((silhouette - mask)**2)

            import pdb
            pdb.set_trace()
            '''
            for coord_kpts_2d in coords_kpts_2d:
                K = targets['K'][0]  # (3,3)
                pts3d_gt = targets['kpts_3d'][0][0]  # (9,3)

                camera_proj_mat_np = np.array(
                    [[K[0, 0] / K[0, 2]], [K[1, 1] / K[1, 2]], [-1]],
                    dtype=np.float32)
                camera_proj_mat = torch.from_numpy(camera_proj_mat_np).type_as(
                    K)

                P_out = self.bpnp(coord_kpts_2d, pts3d_gt, K)
                Rco = angle_axis_to_rotation_matrix(P_out[:, :3])[0]
                mat_co = torch.eye(4).type_as(images)
                mat_co[:3, :3] = Rco
                mat_co[:3, 3] = P_out[0, 3:]
                cam_rot, cam_trans = make_camera_mat_from_mat(mat_co)
                camera_r_param = cam_rot.type_as(images)
                camera_t_param = cam_trans.type_as(images)
                camera_params = [
                    camera_r_param[None], camera_t_param[None], camera_proj_mat
                ]
                predictions, silhouette, _ = self.renderer(
                    points=[
                        self.vertices.type_as(images),
                        self.faces.type_as(images).long()
                    ],
                    camera_params=camera_params,
                    uv_bxpx2=self.uv.type_as(images),
                    texture_bx3xthxtw=self.texture.type_as(images),
                    lightdirect_bx3=self.tflight_bx3.tyep_as(images),
                    material_bx3x3=self.tfmat.type_as(images),
                    shininess_bx1=self.tfshi.tyep_as(images))
                loss += torch.mean((predictions - images[idx])**2)
                loss += torch.mean((silhouette - mask)**2)
            '''
            # mask_np = ten2num(mask[idx])
            # heatmap_th = 0.9
            # heatmap_data = ten2num(
            #     heatmap_outputs[idx, 0] > heatmap_th) * mask_np
            # heatmap_nonzero_coord = np.nonzero(heatmap_data)
            # heatmap_nonzero_coord = np.array(heatmap_nonzero_coord).transpose()
            # # vertex_outputs: torch.Size([2, 16, 480, 640])
            # coords_kpts_2d = extract_keypoints_from_coords(
            #     heatmap_nonzero_coord, ten2num(vertex_outputs[idx].detach()))
            # # heatmap_nonzero_coord
            import pdb
            pdb.set_trace()

        losses, individual_losses = self.calculate_loss(
            segmentation_outputs, heatmap_outputs, vertex_outputs, targets)

        logs = {'train_total_loss': losses}
        for name, individiual_loss in individual_losses.items():
            logs['train_' + name] = individiual_loss
        return {'loss': losses, 'log': logs}

    def calc_detection_metric(self, correct_peak_num, target_num, target):
        detection_metric = {}
        detection_metric["TP"] = torch.tensor(min(correct_peak_num,
                                                  target_num)).type_as(target)
        detection_metric["FP"] = torch.tensor(
            max(target_num - correct_peak_num, 0)).type_as(target)
        detection_metric["FN"] = torch.tensor(
            max(correct_peak_num - target_num, 0)).type_as(target)
        return detection_metric

    def calc_vote_pose_metric(self, peaks, mask, vector_vertex,
                              model_file_path, targets):
        vote_kpts_2d = extract_keypoints_peakvoting(mask, vector_vertex, peaks)
        vote_metric = self.metrics(vote_kpts_2d, mask, targets, "vote",
                                   model_file_path, self.params,
                                   hydra.utils.get_original_cwd())
        return vote_metric

    def calc_center_pose_metric(self, peaks, mask, vertex_outputs,
                                model_file_path, targets):
        vertex = ten2num(vertex_outputs[0])
        peaks_kpts_2d = extract_keypoints_peaks(peaks, vertex)
        center_metric = self.metrics(peaks_kpts_2d, mask, targets, "center",
                                     model_file_path, self.params,
                                     hydra.utils.get_original_cwd())
        return center_metric

    def calc_peak_and_target_num(self, peaks, targets, heatmap_th=0.3):
        # get mask which include the value of class e.g. 0,1,2...
        target_heatmap = torch.sum(targets['heat_maps'][0], dim=0)
        # check if detection number is equal of ground truth
        correct_peak_num = 0
        for peak in peaks:
            correct_peak_num += int(
                target_heatmap[peak[0], peak[1]] > heatmap_th)
        target_num = sum([
            1 if target_cls[0] != '' else 0 for target_cls in targets['clses']
        ])
        return correct_peak_num, target_num

    def calc_metric(self, peaks, mask, vector_vertex, vertex_outputs, targets):
        peak_num, target_num = self.calc_peak_and_target_num(peaks, targets)
        model_file_path = hydra.utils.to_absolute_path(
            self.params.general.model_path)
        detection_metric = self.calc_detection_metric(peak_num, target_num,
                                                      vector_vertex)
        center_metric, vote_metric = None, None
        if peak_num == target_num:
            center_metric = self.calc_center_pose_metric(
                peaks, mask, vertex_outputs, model_file_path, targets)
            vote_metric = self.calc_vote_pose_metric(peaks, mask,
                                                     vector_vertex,
                                                     model_file_path, targets)
        return detection_metric, center_metric, vote_metric

    def validation_step(self, batch, batch_idx):
        """Do the work necessary for a single validation step."""
        images, targets = batch
        segmentation_outputs, heatmap_outputs, vertex_outputs = self.forward(
            images)
        losses, individual_losses = self.calculate_loss(
            segmentation_outputs, heatmap_outputs, vertex_outputs, targets)

        mask = torch.argmax(
            segmentation_outputs[:, :self.params.general.class_num], dim=1)
        vector_vertex = segmentation_outputs[:, self.params.general.class_num:]

        logs = {'val_loss': losses}
        for name, individiual_loss in individual_losses.items():
            logs['val_' + name] = individiual_loss

        peaks = extract_peaks_from_centroid(ten2num(heatmap_outputs[0, 0]))
        detection_metric, center_metric, vote_metric = self.calc_metric(
            peaks, mask, vector_vertex, vertex_outputs, targets)
        logs.update(detection_metric)
        if center_metric is not None:
            logs.update(center_metric)
        if vote_metric is not None:
            logs.update(vote_metric)

        if self.params.wandb.use_wandb:
            self.wandb_visualization(images, targets, peaks, mask,
                                     vector_vertex, heatmap_outputs,
                                     vertex_outputs, center_metric,
                                     vote_metric, batch_idx)
        return logs

    def validation_epoch_end(self, outputs):
        """Aggregate validation results."""
        logs = {}
        avg_loss_out = None

        def _get_outputs(outputs, key):
            return [output[key] for output in outputs if key in output]

        all_keys = []
        for output in outputs:
            all_keys += list(output.keys())
        # Aggregate Losses.
        for key in all_keys:
            stacked_loss = _get_outputs(outputs, key)
            if len(stacked_loss) > 0:
                avg_loss = torch.stack(stacked_loss).mean()
                logs[key] = avg_loss
                if avg_loss_out is None:
                    avg_loss_out = avg_loss
        return {
            'val_loss': avg_loss_out,
            'log': logs,
        }

    # TODO(taku): add test configuration based on target task
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        # lr scheduler: https://github.com/PyTorchLightning/pytorch-lightning/issues/98
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.params.training.lr,
            weight_decay=self.params.training.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.params.training.lr_drop)
        return [optimizer], [lr_scheduler]

    def wandb_visualization(self,
                            images,
                            targets,
                            peaks,
                            mask,
                            vector_vertex,
                            heatmap_outputs,
                            vertex_outputs,
                            center_metric,
                            vote_metric,
                            batch_idx,
                            heatmap_th=0.2):
        mask_labels = {}
        for i, class_label in enumerate(self.params.general.class_labels):
            mask_labels[i] = class_label
        wandb_masks = {}
        if batch_idx % self.params.validation.batch_interval == 0:
            image = images[0]
            mask_np = ten2num(mask[0])
            wandb_masks[f"pred_mask{batch_idx}"] = {
                "mask_data": mask_np,
                "class_labels": mask_labels
            }
            # heatmap_th = 0.8 -> 111, dis=0.5
            # heatmap_th = 0.9 -> 25, dis=0.5
            heatmap_data = ten2num(
                heatmap_outputs[0, 0] > heatmap_th) * mask_np
            wandb_masks[f"pred_heatmap{batch_idx}"] = {
                "mask_data": heatmap_data,
                "class_labels": mask_labels
            }
            if len(peaks) > 0:
                displacement_vertex_np = ten2num(vertex_outputs[0])
                kpts_2d_center = extract_keypoints_peaks(
                    peaks, displacement_vertex_np)
                image = draw_keypoints(image, kpts_2d_center)
                kpts_2d_vote = extract_keypoints_peakvoting(
                    mask, vector_vertex, peaks)
                image = draw_keypoints(image, kpts_2d_vote, (0, 255, 255))

            image = draw_keypoints(image, ten2num(targets['kpts_2d'][0]),
                                   (255, 0, 0))
            if center_metric is not None:
                image = draw_text(
                    image, 'center_adi_value:' +
                    str(center_metric['center_metric_adi'].numpy()))
            if vote_metric is not None:
                image = draw_text(
                    image, 'vote_adi_value:' +
                    str(vote_metric['vote_metric_adi'].numpy()), (10, 469))

            global_step = 0
            if self.trainer:
                global_step = self.trainer.global_step

            wandb.log({
                f"image{global_step}_{batch_idx}": [
                    wandb.Image(images[0], masks=wandb_masks),
                    wandb.Image(image)
                ]
            })


def set_seed(seed: int = 666):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(cfg: DictConfig):
    set_seed(cfg.training.seed)

    if cfg.general.checkpoint == '':
        model = ObjectPoseEstimationModel(cfg)
    else:
        # checkpoint loading:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2550
        model = ObjectPoseEstimationModel(cfg)
        checkpoint_path = hydra.utils.to_absolute_path(cfg.general.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

    # checkpointer = pl.callbacks.ModelCheckpoint(save_top_k=3,
    #                                             filepath=cfg.general.file_path)
    checkpointer = pl.callbacks.ModelCheckpoint(filepath=cfg.general.file_path)
    if not cfg.wandb.use_wandb:
        trainer = pl.Trainer(checkpoint_callback=checkpointer, **cfg.trainer)
    else:
        from pytorch_lightning.loggers import WandbLogger
        import datetime
        log_name = str(datetime.datetime.now())
        wandb_logger = WandbLogger(name=log_name,
                                   project=cfg.wandb.project_dir)
        trainer = pl.Trainer(checkpoint_callback=checkpointer,
                             logger=wandb_logger,
                             **cfg.trainer)
    trainer.fit(model)


# @hydra.main(config_path="pose_config/config.yaml", strict=False)
@hydra.main(config_path="pose_config/config_domain.yaml", strict=False)
def run_model(cfg: DictConfig) -> None:
    print(cfg.pretty())
    run(cfg)


if __name__ == "__main__":
    run_model()
