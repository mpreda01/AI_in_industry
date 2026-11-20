import numpy as np

_MAX_BOXES_PER_CHUNK = 64


def enlarge_boxes(boxes, delta):
    '''
    Enlarge (N,7) boxes (x,y,z,h,w,l,ry) by delta meters on every side, keeping the box center fixed.
    (x,y,z) is the center of the bottom face and y points down, so the bottom face moves down by delta.
    '''
    boxes = np.array(boxes, dtype=np.float64).reshape(-1, 7)
    boxes[:, 1] += delta
    boxes[:, 3:6] += 2 * delta
    return boxes


def points_in_boxes(xyz, boxes):
    '''
    xyz (P,3) points, boxes (N,7) (x,y,z,h,w,l,ry) in the rectified camera frame
    output (N,P) boolean mask, True where point p lies inside box n (borders included)
    '''
    xyz = np.asarray(xyz, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    mask = np.zeros((len(boxes), len(xyz)), dtype=bool)
    for start in range(0, len(boxes), _MAX_BOXES_PER_CHUNK):          # bounded memory
        b = boxes[start:start + _MAX_BOXES_PER_CHUNK]
        rel = xyz[None, :, :] - b[:, None, :3]                         # (n,P,3) relative to the bottom center
        c, s = np.cos(b[:, 6])[:, None], np.sin(b[:, 6])[:, None]
        # Rotate into the box frame (inverse of the rotation around y used in label2corners)
        local_x = c * rel[..., 0] - s * rel[..., 2]
        local_z = s * rel[..., 0] + c * rel[..., 2]
        local_y = rel[..., 1]
        mask[start:start + len(b)] = ((np.abs(local_x) <= b[:, 5:6] / 2) &
                                      (np.abs(local_z) <= b[:, 4:5] / 2) &
                                      (local_y <= 0) & (local_y >= -b[:, 3:4]))
    return mask


def sample_indices(indices, max_points):
    '''Return exactly max_points indices: a random subset if there are more, all of them plus random repeats if fewer.'''
    if len(indices) >= max_points:
        return np.random.choice(indices, max_points, replace=False)
    repeats = np.random.choice(indices, max_points - len(indices), replace=True)
    return np.concatenate([indices, repeats])


def roi_pool(pred, xyz, feat, config):
    '''
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie
       in each enlarged bounding box. Each ROI should contain exactly M=512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (N,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K',7)
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    '''
    pred = np.asarray(pred).reshape(-1, 7)
    M = config['max_points']
    inside = points_in_boxes(xyz, enlarge_boxes(pred, config['delta']))     # (N,P)

    valid = inside.any(axis=1)                       # boxes without a single point are discarded
    pooled_idx = np.stack([sample_indices(np.flatnonzero(m), M) for m in inside[valid]]) if valid.any() \
                 else np.zeros((0, M), dtype=int)

    valid_pred = pred[valid]                         # the original (not enlarged) boxes
    pooled_xyz = xyz[pooled_idx]                     # (K',M,3)
    pooled_feat = feat[pooled_idx]                   # (K',M,C)
    return valid_pred, pooled_xyz, pooled_feat
