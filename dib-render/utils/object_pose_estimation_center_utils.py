"""Utilities specific to object pose estimation training and evaluation."""

import numpy as np
import torch
import cv2
from skimage.feature import peak_local_max


def ten2num(input_tensor, ttype=torch.FloatTensor):
    return input_tensor.type(ttype).numpy()


def get_vertex_displacement_field(kpt_2d, heatmap, threshold=0.3):
    """Get vertex displacement field from keypoints and heatmaps.
    Args:
        kpt_2d (numpy.array): (8, 2)
        heatmap (numpy.array): (height, width)
        threshold (float, optional): Defaults to 0.3.
    Returns:
        (numpy.array): (height, width, 16)
    """
    assert kpt_2d.shape[0] == 8
    assert kpt_2d.shape[1] == 2
    height, width = heatmap.shape
    mask = heatmap > threshold
    coords = np.indices([height, width]).transpose(
        (1, 2, 0)).astype(np.float32)
    disp_field = np.zeros((height, width, 2), dtype=np.float32)
    disp_fields = np.zeros((height, width, 16), dtype=np.float32)
    # For each vertex compute the displacement field.
    for i in range(8):
        disp_field[:, :, :] = 0.0
        vertex_point = np.array([kpt_2d[i][1], kpt_2d[i][0]], dtype=np.float32)
        disp_field[mask] = coords[mask] - vertex_point
        # Normalize by height and width
        norm_height = 1.0 - (disp_field[mask][:, 0] + float(height)) / (2.0 *
                                                                        height)
        norm_width = 1.0 - (disp_field[mask][:, 1] + float(width)) / (2.0 *
                                                                      width)
        disp_field[mask] = np.array([norm_height, norm_width]).T
        disp_fields[:, :, i * 2:i * 2 + 2] = disp_field
    return disp_fields


def get_vertex_vector_field(mask, kpt_2d):
    """Get vertex vector field from keypoints and heatmaps.
    Args:
        mask (numpy.array): (height, width)
        kpt_2d (numpy.array): (9, 2)
    Returns:
        (numpy.array): (height, width, 18)
    """
    assert kpt_2d.shape[0] == 9
    assert kpt_2d.shape[1] == 2
    height, width = mask.shape
    vertex_num = kpt_2d.shape[0]
    inlier_xy = np.argwhere(mask == 1)[:, [1, 0]]
    vertex = kpt_2d[None] - inlier_xy[:, None]
    norm = np.linalg.norm(vertex, axis=2, keepdims=True)
    norm[norm < 1e-3] += 1e-3
    vertex_norm = vertex / norm
    vertex_fields = np.zeros([height, width, vertex_num, 2], np.float32)
    vertex_fields[inlier_xy[:, 1], inlier_xy[:, 0]] = vertex_norm
    vertex_fields = np.reshape(vertex_fields, [height, width, vertex_num * 2])
    return vertex_fields


def candidate_keypoints_np(vectors, coords, hypo_num=128):
    """Calculate the candidate key points using vertex vector field.
    Args:
        vectors (numpy.array): (inlier_num, 9, 2)
        coords (numpy.array): (inlier_num, 2)
        hypo_num (int, optional): Defaults to 128.
    Returns:
        (numpy.array): (hypo_num, 9, 2)
    """
    vertex_num = vectors.shape[1]
    target_num = vectors.shape[0]
    vectors = vectors[:, :, [1, 0]]  # [inlier_num, 9, 2]
    vectors[:, :, 1] *= -1

    # select 2 set of random points which size is hypo_num and
    # get vertex vector field, these coordinats.
    selected_id1 = np.random.randint(target_num, size=hypo_num)
    vectors_mat1 = vectors[selected_id1, :, :]
    selected_id2 = np.random.randint(target_num, size=hypo_num)
    vectors_mat2 = vectors[selected_id2, :, :]
    coords_mat1 = np.repeat(coords[selected_id1][:, None], vertex_num, axis=1)
    coords_mat2 = np.repeat(coords[selected_id2][:, None], vertex_num, axis=1)

    # calculate intersection points.
    # refer [https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection]
    numerator_mat1 = np.repeat(np.sum(vectors_mat1 * coords_mat1, 2)[...,
                                                                     None],
                               2,
                               axis=2)
    numerator_mat2 = np.repeat(np.sum(vectors_mat2 * coords_mat2, 2)[...,
                                                                     None],
                               2,
                               axis=2)
    denominator_mat = np.repeat(
        (vectors_mat1[..., 0] * vectors_mat2[..., 1] -
         vectors_mat1[..., 1] * vectors_mat2[..., 0])[..., None],
        2,
        axis=2)
    hypo_pts = (vectors_mat2[:, :, [1, 0]] * numerator_mat1 -
                vectors_mat1[:, :, [1, 0]] * numerator_mat2) / denominator_mat
    hypo_pts[:, :, 1] *= -1
    return hypo_pts


def voting_inliers_np(vectors, coords, hypo_pts, inlier_thresh=0.99):
    """Get high voting rate inlier field which has hypo_num layers as bool.
    Args:
        vectors (numpy.array): (inlier_num, 9, 2)
        coords (numpy.array): (inlier_num, 2)
        hypo_pts (numpy.array): (hypo_num, 9, 2)
        (float, optional): Defaults to 0.99.
    Returns:
        (numpy.array): (hypo_num, 9, inlier_num)
    """
    hypo_num = hypo_pts.shape[0]
    vertex_num = hypo_pts.shape[1]
    target_num = vectors.shape[0]

    # calculate the cosine value between the vertex vector and
    # the vector which is calculated by candidate point and pixel coordinate
    vectors_norm = np.power(vectors, 2)[..., 0] + np.power(
        vectors, 2)[..., 1]  # because np.sum is slow
    vectors_norm = np.sqrt(vectors_norm).transpose(1, 0)

    hypo_mat = np.repeat(hypo_pts[:, :, None], target_num, axis=2)
    coords_mat = np.repeat(coords[None], vertex_num, axis=0)
    coords_mat = np.repeat(coords_mat[None], hypo_num, axis=0)

    diff_norm = hypo_mat - coords_mat
    diff_norm_sq = diff_norm * diff_norm  # because np.power is slow
    diff_norm_sq = diff_norm_sq[..., 0] + diff_norm_sq[..., 1]
    diff_norm_sq = np.sqrt(diff_norm_sq)

    vectors_mat = np.repeat(vectors[None], hypo_num,
                            axis=0).transpose(0, 2, 1, 3)
    diff_mat = hypo_mat - coords_mat
    angle_sum = vectors_mat * diff_mat
    angle_sum = angle_sum[..., 0] + angle_sum[..., 1]
    angle_mat = angle_sum / (vectors_norm * diff_norm_sq)
    # if consine value is over inlier_thresh, this point will be True.
    inliers = (angle_mat > inlier_thresh)
    return inliers


def normal_equation_np(normal, bv):
    """Solve normal equation.
    Args:
        normal (numpy.array): e.g. 9 vertexes case -> (9, inlier_num, 2)
        bv (numpy.array): e.g. 9 vertexes case -> (9, inlier_num)
    Returns:
        (numpy.array): (9, 2, 1)
    """
    ata = normal.transpose(0, 2, 1) @ normal
    bs = bv[..., None]
    atb = np.sum(normal * bs, 1)
    inverse_ata = np.linalg.pinv(ata)
    atb = atb[..., None]
    ans = inverse_ata @ atb
    return ans


def voting_keypoints_np(mask,
                        vertex,
                        hypo_num=128,
                        inlier_thresh=0.999,
                        confidence=0.7,
                        max_iter=20,
                        min_num=5,
                        max_num=30000):
    """Get vertex and center keypoint 2d location.
    This function try to get x,y coordinate of 2d vertex and center location
    by ransac + voting using normal vector field which point toward
    from each pixel to vertex and center location.
    Args:
        mask (numpy.array): [1, height, width]
        vertex (numpy.array): [1, height, width, 9, 2]
    Returns:
        (numpy.array): (1, 9, 2)
    """
    assert mask.shape[0] == 1
    assert vertex.shape[0] == 1
    assert vertex.shape[3] == 9
    assert vertex.shape[4] == 2
    vertex_num = vertex.shape[3]
    bool_mask = (mask[0]).astype(bool)
    foreground_num = np.sum(bool_mask)

    # if too few inliers, just skip it
    if foreground_num < min_num:
        return np.zeros((1, vertex_num, 2))

    # if too many inliers, randomly down sample it
    if foreground_num > max_num:
        selection = np.random.rand(bool_mask.shape[0], bool_mask.shape[1])
        selected_mask = \
            (selection < (max_num / foreground_num.astype(float))).astype(bool)
        bool_mask *= selected_mask

    # coordinate of inliers which is segmentation area
    coords = np.stack(np.nonzero(bool_mask),
                      axis=1).astype(float)[:, [1, 0]]  # (inlier_num, 2)
    target_num = coords.shape[0]
    target_vertex = vertex[0][bool_mask, :, :]  # (height, width, 9, 2)
    win_ratio = np.zeros((vertex_num))
    win_pts = np.zeros((vertex_num, 2))

    # get the high confidence inlier pixel which point toward vertex by voting
    # if iteration is over max_iter or it get high confidence, pass the loop
    cur_iter = 0
    while True:
        hypo_pts = candidate_keypoints_np(target_vertex, coords,
                                          hypo_num)  # (hypo_num, 9, 2)
        cur_inlier = voting_inliers_np(
            target_vertex, coords, hypo_pts,
            inlier_thresh)  # (hypo_num, 9, inlier_num)

        # find max key point
        cur_inlier_counts = np.sum(cur_inlier, 2)  # (hypo_num, 2)
        cur_win_counts = np.max(cur_inlier_counts, 0)  # (9,)
        cur_win_id = np.argmax(cur_inlier_counts, 0)  # (9,)
        cur_win_pts = hypo_pts[cur_win_id, np.arange(vertex_num)]  # (9,2)
        cur_win_ratio = cur_win_counts.astype(float) / target_num  # (9,)

        # update best point
        larger_mask = win_ratio < cur_win_ratio
        win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]  # (9,2)
        win_ratio[larger_mask] = cur_win_ratio[larger_mask]  # (9,)

        cur_iter += 1
        cur_min_ratio = np.min(win_ratio)
        # TODO(taku): figure out the good confidence value
        if cur_min_ratio > confidence or cur_iter > max_iter:
            break

    # solve the normal equation for getting intersection points
    # using normal vector from each inlier pixel to 8 vertex and center
    # if you want to know more, refer [https://arxiv.org/abs/1812.11788]
    inlier = voting_inliers_np(target_vertex, coords, win_pts[None],
                               inlier_thresh)[0][...,
                                                 None]  # (9, inlier_num,1)
    normal = np.zeros(target_vertex.shape)  # (inlier_num, 9, 2)
    normal[:, :, 0] = target_vertex[:, :, 1]
    normal[:, :, 1] = -target_vertex[:, :, 0]
    normal = normal.transpose(1, 0, 2)  # (9, inlier_num, 2)
    normal = normal * inlier
    bv = np.sum(normal * coords[None], 2)  # (9, inlier_num)
    win_pts = normal_equation_np(normal, bv)  # (9, 2, 1)
    return win_pts[None, :, :, 0]  # (1, 9, 2)


# TODO(taku): consider the better way to find the peak area instead of fix one
def decode_keypoints_peakmask(peaks, vectors, seg_mask, peak_area=30):
    """Decode keypoints using peaks and mask area
    Args:
        peaks(numpy.array): [peak_num, 2]
        vectors(torch.tensor): [1, 18, height, width]
        seg_mask(numpy.array): [1, 480, 640]
        peak_area (int, optional): Defaults to 30.
    Returns:
        (numpy.array): (peak_num, 9, 2)
    """
    '''
    peaks(numpy.array): [peak_num, 2]
    vectors(torch.tensor): [1, 18, height, width]
    seg_mask(numpy.array): [1, 480, 640]
    '''
    assert vectors.shape[0] == 1
    assert vectors.shape[1] == 18
    assert seg_mask.shape[0] == 1
    vertex = vectors.permute(0, 2, 3, 1)
    batch, height, width, vertex_num = vertex.shape
    vertex = vertex.view(batch, height, width, vertex_num // 2, 2)
    np_vertex = ten2num(vertex)

    kpt_2ds = np.zeros((len(peaks), 9, 2), dtype=np.int)
    for i, peak in enumerate(peaks):
        # peak area value is important of accuracy from voting.
        peak_mask = np.zeros((1, height, width))
        min_x = max(0, peak[1] - peak_area)
        max_x = min(width, peak[1] + peak_area)
        min_y = max(0, peak[0] - peak_area)
        max_y = min(height, peak[0] + peak_area)
        peak_mask[:, min_y:max_y, min_x:max_x] = 1.0
        overlap_mask = peak_mask * seg_mask
        # TODO(taku): maybe should cut the first dim of kpt_2d output
        kpt_2d = voting_keypoints_np(overlap_mask,
                                     np_vertex,
                                     hypo_num=128,
                                     inlier_thresh=0.99,
                                     max_num=100)
        kpt_2ds[i] = kpt_2d[0].astype(np.int)
    return kpt_2ds


def draw_2d_keypoint(img, kpt_2d, bgrv=(0, 255, 0)):
    """Draw 8 keypoint on numpy image.
    Args:
        img (numpy.array): (height, width, 3)
        kpt_2d (numpy.array): (kpt_num, 2) kpt_num>=8
        bgrv (tuple, optional): Defaults to (0, 255, 0).
    Returns:
        (numpy.array): (height, width, 3)
    """
    permute1 = [0, 1, 3, 2, 0, 4, 6, 2]
    permute2 = [5, 4, 6, 7, 5, 1, 3, 7]
    for i in range(7):
        cv2.line(img,
                 tuple(kpt_2d[permute1[i]]),
                 tuple(kpt_2d[permute1[i + 1]]),
                 bgrv,
                 thickness=2)
        cv2.line(img,
                 tuple(kpt_2d[permute2[i]]),
                 tuple(kpt_2d[permute2[i + 1]]),
                 bgrv,
                 thickness=2)
    return img


def extract_peaks_from_centroid(centroid_heatmap, min_distance=5, th_abs=0.3):
    """Extract peak values from heatmap.
    Args:
        centroid_heatmap (numpy.array): (height, width)
        min_distance (int, optional): Defaults to 5.
        th_abs (float, optional): Defaults to 0.3.
    Returns:
        (numpy.array): (peak_num, 2)
    """
    peaks = peak_local_max(centroid_heatmap,
                           min_distance=min_distance,
                           threshold_abs=th_abs)

    return peaks


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


def extract_vertices_from_peaks(peaks, vertex_fields, img, scale_factor=1):
    """Extract keypoints from peaks and vertex displacement field.
    Args:
        peaks (numpy.array): (peak_num, 2)
        vertex_fields (numpy.array): (height, width, 16)
        img (numpy.array): (height, width, 16)
        scale_factor (int, optional): Defaults to 1.
    Returns:
        (numpy.array): (peak_num, 8, 2)
    """
    assert peaks.shape[1] == 2
    assert vertex_fields.shape[2] == 16
    height, width = img.shape[0:2]
    # denormalize using height and width
    vertex_fields[:, :, ::2] = (1.0 - vertex_fields[:, :, ::2]) * (
        2 * height) - height
    vertex_fields[:, :,
                  1::2] = (1.0 - vertex_fields[:, :, 1::2]) * (2 *
                                                               width) - width
    vertices = np.zeros(
        (len(peaks), vertex_fields.shape[2] // 2, peaks.shape[1]))
    for i, peak in enumerate(peaks):
        vertices[i] = get_bbox_vertices_from_vertex(vertex_fields,
                                                    peak,
                                                    scale_factor=scale_factor)
    return vertices


def extract_keypoints_peakvoting(mask, vector, peaks, peak_area=30):
    """Extract keypoints from vertex and center vector field.
    Args:
        mask(torch.tensor): (1, height, width)
        vector(torch.tensor): (1, 18, width, height)
        peaks(numpy.array): (peak_num, 2)
        peak_area (int, optional): Defaults to 30.
    """
    assert mask.shape[0] == 1
    assert vector.shape[0] == 1
    assert vector.shape[1] == 18
    np_mask = ten2num(mask)
    conc_mask = np.zeros(
        (np_mask.shape[0], np_mask.shape[1], np_mask.shape[2]))
    conc_mask[np_mask > 0] = 1
    kpts_2d = decode_keypoints_peakmask(peaks, vector, conc_mask, peak_area)
    return kpts_2d  # numpy array


def extract_keypoints_peaks(peaks, vertex):
    """Extract keypoints from peaks and vertex displacement field.
    Args:
        peaks(numpy.array): (peak_num, 2)
        vertex(numpy.array): (16, height, width)
    """
    # TODO(taku): refactor the below function to more simple
    kpts_2d = extract_vertices_from_peaks(peaks, vertex.transpose(1, 2, 0),
                                          vertex.transpose(1, 2, 0), 1)
    # Adjust the center point using peak value
    kpts_2d = kpts_2d - \
        (np.sum(np.array(kpts_2d), axis=1) / 8 - peaks)[:, None]
    kpts_2d_with_center = np.concatenate([kpts_2d, peaks[:, None, :]], 1)
    return kpts_2d_with_center[:, :, [1, 0]].astype(int)  # numpy array


def draw_keypoints(image, kpts_2d, color=(0, 0, 255)):
    """Draw keypoints on the torch tensor image
    Args:
        image (torch.tensor): (channel, height, width)
        kpt_2d (numpy.array): (kpt_num, 2) kpt_num>=8
    """
    np_image = ten2num(image.permute(1, 2, 0) * 255).astype(np.uint8).copy()
    for kpt_2d in kpts_2d:
        debug_image = draw_2d_keypoint(np_image, kpt_2d, color)
    debug_image = torch.from_numpy(debug_image / 255.0).type_as(image)
    debug_image = debug_image.permute(2, 0, 1)
    return debug_image


def draw_text(image, text, loc=(10, 440)):
    """Draw the text on the torch tensor image
    Args:
        image (torch.tensor): (channel, height, width)
        text (str): draw the text on the image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    np_image = ten2num(image.permute(1, 2, 0) * 255).astype(np.uint8).copy()
    cv2.putText(np_image, text, loc, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    debug_image = torch.from_numpy(np_image / 255.0).type_as(image)
    debug_image = debug_image.permute(2, 0, 1)
    return debug_image


def project(points_3d, intrinsics, pose):
    points_3d = np.dot(points_3d, pose[:, :3].T) + pose[:, 3:].T
    points_3d = np.dot(points_3d, intrinsics.T)
    points_2d = points_3d[:, :2] / points_3d[:, 2:]
    return points_2d


def debug_pose_image(img, kpt_3d, camera_info_K, pose):
    corner_2d = project(kpt_3d, camera_info_K, pose).astype(np.uint8)
    permute1 = [0, 1, 3, 2, 0, 4, 6, 2]
    permute2 = [5, 4, 6, 7, 5, 1, 3, 7]
    for i in range(7):
        cv2.line(img,
                 tuple(corner_2d[permute1[i]]),
                 tuple(corner_2d[permute1[i + 1]]), (0, 255, 0),
                 thickness=2)
        cv2.line(img,
                 tuple(corner_2d[permute2[i]]),
                 tuple(corner_2d[permute2[i + 1]]), (0, 255, 0),
                 thickness=2)
    return img
