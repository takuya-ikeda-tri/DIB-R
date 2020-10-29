"""Metrics used for object pose estimation training and validation"""
import os
import torch
from utils.image_utils import solve_pnp
import open3d as o3d
import numpy as np
import yaml
from scipy import spatial


def load_models_info(models_path, base_path='.', scale=1.0):
    """Load mesh model and return vertices as dictionary.
    Args:
        models_path (str): path of models.yaml
        base_path (str, optional): path of base dir. Defaults to '.'.
        scale (float, optional): Defaults to 1.0.
    Returns:
        (dict): {key: vertices}
    """
    def _get_xyz(model_path):
        mesh = o3d.io.read_triangle_mesh(model_path)
        return np.asarray(mesh.vertices)

    with open(models_path) as file:
        models = yaml.load(file)

    model_xyz = {}
    for key in models:
        model_path = os.path.join(base_path, models[key]['path'])
        model_xyz[str(key)] = _get_xyz(model_path) * scale
    return model_xyz


def add_metrics(model_xyz, pose_pred, pose_gt):
    """Calculate the average distance of distinguishable model points(ADD) metric.
    reference: https://arxiv.org/abs/1711.00199
    Args:
        model_xyz (numpy.array): (point_num, 3)
        pose_pred (numpy.array): (3, 4) transform matrix
        pose_gt (numpy.array): (3, 4) transform matrix
    Returns:
        (numpy.array): add value
    """
    add_pred = np.dot(model_xyz, pose_pred[:, :3].T) + pose_pred[:, 3]
    add_gt = np.dot(model_xyz, pose_gt[:, :3].T) + pose_gt[:, 3]
    # l2 norm -> np.linalg.norm(x, axis=-1)
    add = np.mean(np.linalg.norm(add_pred - add_gt, axis=-1))
    return add


# TODO(taku): should change the variable name
def adi_metrics(model_xyz, pose_pred, pose_gt):
    """Calculate the average distance of indistinguishable model points(ADI)
    metric.
    reference: https://arxiv.org/abs/1711.00199
    Args:
        model_xyz (numpy.array): (point_num, 3)
        pose_pred (numpy.array): (3, 4) transform matrix
        pose_gt (numpy.array): (3, 4) transform matrix
    Returns:
        (numpy.array): adi value
    """
    adi_pred = np.dot(model_xyz, pose_pred[:, :3].T) + pose_pred[:, 3]
    adi_gt = np.dot(model_xyz, pose_gt[:, :3].T) + pose_gt[:, 3]
    nn_index = spatial.cKDTree(adi_pred)
    nn_dists, _ = nn_index.query(adi_gt, k=1)
    adi = nn_dists.mean()
    return adi


# TODO(taku): please consider to use TIDE
# https://github.com/dbolya/tide
def iou_metrics(outputs, labels):
    """Calculate iou metric.
    Args:
        outputs (torch.tensor): (batch_size, height, width)
        labels (torch.tensor): (batch_size, height, width)
    Returns:
        (torch.tensor): iou value
    """
    intersection = (outputs * labels).float().sum((1, 2))
    union = ((outputs + labels) > 0) * 1.0
    union = union.sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def ten2num(input_tensor):
    return input_tensor.type(torch.FloatTensor).numpy()


# TODO(taku): this is icp version, please refacter it
class ObjectPoseEstimationMetrics(object):
    """ Torch object pose estimation metrics."""

    def __call__(self,
                 kpts_2d,
                 mask,
                 targets,
                 name,
                 models_path,
                 params,
                 base_path='.'):
        """Call metric module.
        Args:
            kpts_2d (numpy.array): (kpt_num, 9, 2)
            mask (torch.tensor): (batch_size, height, width)
            targets (torch.tensor): data from data_loader
            targets['kpt_2d'] (torch.tensor): (batch_size, kpt_num, 9, 2)
            name (str): metric tag name
            models_path (str): path of models.yaml
            base_path (str, optional): path of models.yaml. Defaults to '.'.
            model_scale (float, optional): Defaults to 1.0.
        Returns:
            (dict): iou, add, adi value.
        """
        if params.general.model_scale:
            model_scale = params.general.model_scale
        else:
            model_scale = 1.0
        target_keys = params.general.eval_targets

        # IoU Metric
        # TODO(taku): please check the value of targets['mask_class']
        # if output is wrong
        # This ids should be 0,1,2,3... because loss func convert 1,4->1,2
        metric_iou = 0
        class_ids = torch.unique(targets['mask_class'])
        for i, class_id in enumerate(class_ids[1:]):
            mask_pre = mask == class_id
            mask_target = targets['mask_class'] == class_id
            metric_iou += iou_metrics(mask_pre, mask_target)
        metric_iou /= len(class_ids[1:])

        # ADD/ADI Metric(assume peaks number is correct)
        metric_add = 0
        metric_adi = 0
        count = 0
        centers_target = targets['kpts_2d'][0][:, 8]
        centers_pre = kpts_2d[:, 8]
        kd_tree = spatial.KDTree(centers_pre)
        for i, center_target in enumerate(centers_target):
            # Check the target is available or not
            clsi = targets['clses'][i][0]
            if clsi not in target_keys:
                continue

            # search the nearest point from ground truth center point
            kpt_2d = kpts_2d[kd_tree.query(ten2num(center_target))[1]]
            pose_pred = solve_pnp(ten2num(targets['kpts_3d'][0][i]), kpt_2d,
                                  ten2num(targets['K'][0]))

            model_xyz = load_models_info(models_path, base_path, model_scale)
            metric_add += add_metrics(model_xyz[clsi], pose_pred,
                                      ten2num(targets["poses"][0, i]))
            metric_adi += adi_metrics(model_xyz[clsi], pose_pred,
                                      ten2num(targets["poses"][0, i]))
            count += 1

        if count != 0:
            metric_add /= count
            metric_adi /= count
        metrics = {}
        metrics[f"{name}_metric_iou"] = torch.as_tensor(metric_iou,
                                                        dtype=torch.float32)
        metrics[f"{name}_metric_add"] = torch.as_tensor(metric_add,
                                                        dtype=torch.float32)
        metrics[f"{name}_metric_adi"] = torch.as_tensor(metric_adi,
                                                        dtype=torch.float32)
        return metrics
