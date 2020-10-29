"""Losses used for object pose estimation training."""

import torch
import torch.nn as nn


# TODO(taku): consider the name of vertex and vote
# two type of vertex is existed, one is from center,
# the other is vetex from each pixel. So, center_vertex and pixel_vertex
# is clear to me.
# TODO(taku): move it to another file
class ObjectPoseEstimationLoss(nn.Module):
    """Torch module for computing object pose estimation loss."""

    def mask_loss(self, src_segmentations, targets):
        """Calculate mask loss of multi-class segmentation
        https://discuss.pytorch.org/t/multiclass-segmentation/54065
        Args:
            src_segmentations (torch.tensor):
                [batch_size, class_size(bg+target), height, width]
            tar_segmentations (torch.tensor):
                [batch_size, height, width](range:[0, class_size-1])
            obj_ids (list):
                list of object ids e.g. [0, 1, 4](0:bg, 1:class1, 4:class4)
        Returns:
            (torch.tensor): mask loss
        """
        mask_class = targets['mask_class']
        # TODO(taku): should consider matching method
        # dataset loader should handle this!?
        # convert mask value e.g. from [0,1,4] to [0,1,2]
        obj_ids = torch.unique(mask_class)
        for i, obj_id in enumerate(obj_ids):
            mask_class[mask_class == obj_id] = i
        tar_segmentations = mask_class.long()

        crit = nn.CrossEntropyLoss()
        loss = crit(src_segmentations, tar_segmentations)
        return loss

    # TODO(taku): align the src and target shape
    def heatmap_loss(self, src_heatmaps, targets):
        """Calculate heatmap loss.
        Args:
            src_heatmaps (torch.tensor): [batch_size, 1, height, width]
            tar_heatmap (torch.tensor): [batch_size, height, width]
        Returns:
            (torch.tensor): heatmap loss
        """
        tar_heatmap = torch.sum(targets['heat_maps'], dim=1)
        crit = nn.MSELoss(reduction='none')
        loss = crit(src_heatmaps[:, 0], tar_heatmap)
        loss = torch.mean(loss)
        return loss

    def vertex_loss(self,
                    src_vertexes,
                    heatmaps,
                    targets,
                    centroid_threshold=0.3,
                    weight=0.0001):
        """Calculate vertex displacement fields loss.
        Args:
            src_targets (torch.tensor): [batch,16,height,width]
            heatmaps (torch.tensor): [batch, 1, height, width]
            mask (torch.tensor): [batch, height, width]
            tar_vertexes (torch.tensor): [batch,16,height,width]
            centroid_threshold (float, optional): Defaults to 0.3.
            weight (float, optional): Defaults to 0.0001.
        Returns:
            (torch.tensor):vertex displacement fields loss.
        """
        # TODO(taku): data loader should align the order of vertex_fields
        tar_vertexes = torch.sum(targets['vertex_displacement_fields'],
                                 dim=1).permute(0, 3, 1, 2)
        mask = heatmaps[:, 0] > centroid_threshold
        src_vertexes = src_vertexes * mask[:, None]
        loss = nn.functional.smooth_l1_loss(
            src_vertexes, tar_vertexes, reduction='sum') * weight
        return loss

    def vote_loss(self, src_vectors, targets):
        """Calculate vertex and center displacement fields loss.
        Args:
            src_vectors: [batch_size, 18, height, width]
            tar_vectors: [batch_size, 18, height, width]
            targets['vertex_vector_fields']: [batch_size, max_obj_num, 18, height, width]
            weight: [batch_size, 1, height, width]
        Returns:
            (torch.tensor): vertex and center vector fields loss.
        """
        # TODO(taku): consider the weight should come from inference or gt
        weight = torch.sum(targets['masks'],
                           dim=1).float().type_as(src_vectors)[:, None]
        src_vectors = src_vectors * weight
        tar_vectors = torch.sum(targets['vertex_vector_fields'],
                                dim=1).type_as(src_vectors) * weight
        loss = nn.functional.smooth_l1_loss(src_vectors,
                                            tar_vectors,
                                            reduction='sum')
        # TODO(taku): consider better way for normlization
        loss = loss / weight.sum() / targets['vertex_vector_fields'].size(2)
        return loss

    # TODO(taku): please consider perpendicular_loss
    def forward(self, segmentations, heatmaps, vertexes, targets, params):
        """Compute loss from (network) output and gt target."""
        mask_loss = self.mask_loss(segmentations[:, :params.general.class_num],
                                   targets)
        heatmap_loss = self.heatmap_loss(heatmaps, targets)
        vertex_loss = self.vertex_loss(vertexes, heatmaps, targets)
        vote_loss = self.vote_loss(segmentations[:, params.general.class_num:],
                                   targets)
        # TODO(taku): add perpendicular loss here
        return mask_loss, heatmap_loss, vertex_loss, vote_loss
