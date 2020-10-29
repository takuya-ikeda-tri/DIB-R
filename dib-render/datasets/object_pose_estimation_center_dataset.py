"""Object Pose Estimation dataset loader."""
import torch.utils.data as torch_data
from pycocotools.coco import COCO
import torchvision.transforms as T
import numpy as np
from PIL import Image
import os
from utils.object_pose_estimation_center_utils import get_vertex_displacement_field
from utils.object_pose_estimation_center_utils import get_vertex_vector_field
from utils.image_utils import get_heatmap_gaussian


# TODO(taku): Add Normalize or so
def build_transform(is_training):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


class ObjectPoseEstimationDataset(torch_data.Dataset):
    """Loads Torch dataset."""
    def __init__(self,
                 ann_file,
                 transform=None,
                 base_path='.',
                 objnum=1,
                 depth_use=False):
        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self.transform = transform
        self.base_path = base_path
        self.max_objnum = objnum
        self.depth_use = depth_use

    def __len__(self):
        return len(self.img_ids)

    def read_data(self, idx):
        img_ids = self.img_ids[idx]
        # load image info
        # TODO(taku): change the file_name to rgb_path
        img_path = os.path.join(
            self.base_path,
            self.coco.loadImgs(int(img_ids))[0]['file_name'])
        img = np.array(Image.open(img_path).convert("RGB"))

        # load annotation info
        ann_ids = self.coco.getAnnIds(imgIds=img_ids)
        anns = self.coco.loadAnns(ann_ids)

        # load mask info
        mask_path = os.path.join(self.base_path, anns[0]['mask_path'])
        mask = np.array(Image.open(mask_path))
        mask_class_path = os.path.join(self.base_path,
                                       anns[0]['mask_class_path'])
        mask_class = np.array(Image.open(mask_class_path))

        if not self.depth_use:
            return img, mask, mask_class, anns
        else:
            # loading depth info
            depth_img_path = self.coco.loadImgs(int(img_ids))[0]['depth_path']
            depth_img_path = os.path.join(
                self.base_path,
                self.coco.loadImgs(int(img_ids))[0]['depth_path'])
            depth_img = np.array(Image.open(depth_img_path), dtype=np.float32)
            return img, mask, mask_class, depth_img, anns

    def __getitem__(self, idx):
        target = {}
        if not self.depth_use:
            img, mask, mask_class, anns = self.read_data(idx)
        else:
            img, mask, mask_class, depth_img, anns = self.read_data(idx)
            target["depth"] = depth_img
            target["depth_scale"] = np.array(anns[0]['depth_scale'])

        # Define target elements(boxes, masks, poses, kpts_2d, kpts_3d,
        # heatmaps, vertex_displacement_fields, vertex_vector_field, clses)
        obj_ids = np.unique(mask)[1:]  # remove the background id
        num_objs = len(obj_ids)
        assert self.max_objnum >= num_objs

        # (taku): max objnum is for solving batch stack issue.
        # you can also solve this by making custom collate_fn.
        boxes = np.zeros((self.max_objnum, 4), dtype=np.float32)
        masks = np.zeros((self.max_objnum, mask.shape[0], mask.shape[1]),
                         dtype=np.int)
        masks[:num_objs] = mask == obj_ids[:, None, None]
        heat_maps = np.zeros((self.max_objnum, mask.shape[0], mask.shape[1]),
                             dtype=np.float32)
        pose = np.array(anns[0]['pose'])
        poses = np.zeros((self.max_objnum, pose.shape[0], pose.shape[1]),
                         dtype=np.float32)
        kpt_2d = np.concatenate([anns[0]['fps_2d'], [anns[0]['center_2d']]],
                                axis=0)
        kpts_2d = np.zeros((self.max_objnum, kpt_2d.shape[0], kpt_2d.shape[1]),
                           dtype=np.float32)
        kpts_3d = np.zeros(
            (self.max_objnum, kpt_2d.shape[0], (kpt_2d.shape[1] + 1)),
            dtype=np.float32)
        vertex_vector_fields = np.zeros((self.max_objnum, 2 * kpt_2d.shape[0],
                                         mask.shape[0], mask.shape[1]),
                                        dtype=np.float32)
        vertex_displacement_fields = np.zeros(
            (self.max_objnum, mask.shape[0], mask.shape[1],
             2 * kpt_2d[:8].shape[0]),
            dtype=np.float32)
        clses = [''] * self.max_objnum

        for i in range(num_objs):
            # TODO(taku): make small func below, such as calc_xyxy_bbox_mask
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes[i] = [xmin, ymin, xmax, ymax]
            heat_map = np.zeros((mask.shape[0], mask.shape[1]),
                                dtype=np.float32)
            box_height = ymax - ymin
            box_width = xmax - xmin
            obj_center = anns[i]['center_2d']
            heat_maps[i] = get_heatmap_gaussian(heat_map, obj_center,
                                                (box_height, box_width))
            poses[i] = np.array(anns[i]['pose'])
            kpts_2d[i] = np.concatenate(
                [anns[i]['fps_2d'], [anns[i]['center_2d']]], axis=0)
            kpts_3d[i] = np.concatenate(
                [anns[i]['fps_3d'], [anns[i]['center_3d']]], axis=0)
            vertex_displacement_fields[i] = get_vertex_displacement_field(
                kpts_2d[i][:8], heat_maps[i], 0.3)
            vertex_vector_fields[i] = get_vertex_vector_field(
                masks[i], kpts_2d[i]).transpose(2, 0, 1)
            # TODO(taku): change the format.
            # e.g. clses = [('1',), ('4',)] -> ['1', '4']
            clses[i] = str(anns[i]['cls'])
        image_id = np.array([idx])

        target["image_id"] = image_id  # none yet
        target["boxes"] = boxes  # none yet
        target["masks"] = masks  # loss
        target["mask_class"] = mask_class.astype(np.int)  # loss, metric
        target["heat_maps"] = heat_maps  # train, loss
        target["poses"] = poses  # metric
        target["kpts_2d"] = kpts_2d  # train, metric
        target["kpts_3d"] = kpts_3d  # metric
        target[
            "vertex_displacement_fields"] = vertex_displacement_fields  # loss
        target["vertex_vector_fields"] = vertex_vector_fields  # loss
        target["clses"] = clses  # train, metric
        target["K"] = np.array(anns[0]['K'])  # metric

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    # TODO(taku): try several augmentation methods
    def augment(self):
        pass
