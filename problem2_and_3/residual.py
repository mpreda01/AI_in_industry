'''
Proposal-relative (residual) box refinement, enabled with `data: residual: true` in the config.

The course template feeds the pooled points in absolute camera coordinates and regresses absolute boxes, so the
network has to re-locate every car from scratch and the stage-1 box only decides which points are pooled.
Here, as in the PointRCNN stage-2 refinement:
  - the pooled points of each proposal are expressed in the proposal's own frame (origin at its bottom center,
    rotated by -ry around the vertical y axis), so every car looks alike wherever it is in the scene;
  - the network regresses a correction of the proposal (encode_boxes) that decode_boxes applies to it, so
    all-zero outputs reproduce the stage-1 proposal.
Correction (7,) of a proposal p towards a box t:
  (dx, dy, dz)  t's bottom center minus p's, in p's frame (meters)
  (dh, dw, dl)  log(t_size / p_size)
  dry           t.ry - p.ry wrapped to [-pi/2, pi/2): a box turned by pi has the same corners, hence the same IoU
The regression and classification losses (utils/task4.py) are unchanged, applied to corrections instead of boxes.
'''
import numpy as np

from dataset import DatasetLoader
from utils.task2 import roi_pool
from utils.task3 import sample_proposals


def wrap_angle(a, period=2 * np.pi):
    '''Wrap angles to [-period/2, period/2).'''
    return (a + period / 2) % period - period / 2


def to_box_frame(vec, boxes):
    '''Rotate (N,...,3) camera-frame vectors into the frame of the (N,7) boxes (same rotation as task2.points_in_boxes).'''
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    c, s = c.reshape((-1,) + (1,) * (vec.ndim - 2)), s.reshape((-1,) + (1,) * (vec.ndim - 2))
    return np.stack([c * vec[..., 0] - s * vec[..., 2],
                     vec[..., 1],
                     s * vec[..., 0] + c * vec[..., 2]], axis=-1)


def from_box_frame(vec, boxes):
    '''Inverse of to_box_frame (rotation used in task1.label2corners).'''
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    c, s = c.reshape((-1,) + (1,) * (vec.ndim - 2)), s.reshape((-1,) + (1,) * (vec.ndim - 2))
    return np.stack([c * vec[..., 0] + s * vec[..., 2],
                     vec[..., 1],
                     -s * vec[..., 0] + c * vec[..., 2]], axis=-1)


def canonicalize(xyz, boxes):
    '''(N,M,3) pooled points -> coordinates in the frame of their (N,7) proposal.'''
    return to_box_frame(xyz - boxes[:, None, :3], boxes).astype(np.float32)


def encode_boxes(boxes, targets):
    '''(N,7) proposals, (N,7) assigned targets -> (N,7) corrections.'''
    boxes, targets = np.asarray(boxes, np.float64), np.asarray(targets, np.float64)
    center = to_box_frame(targets[:, :3] - boxes[:, :3], boxes)
    # frames without ground truth get a dummy target with negative sizes (dataset.get_data): they are never
    # positives, so their correction is unused, but keep it finite
    size = np.nan_to_num(np.log(np.abs(targets[:, 3:6]) / boxes[:, 3:6]))
    ry = wrap_angle(targets[:, 6] - boxes[:, 6], np.pi)
    return np.concatenate([center, size, ry[:, None]], axis=1).astype(np.float32)


def decode_boxes(boxes, deltas):
    '''(N,7) proposals, (N,7) corrections (numpy or torch) -> (N,7) boxes, inverse of encode_boxes.'''
    to_np = lambda a: a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)
    boxes, deltas = to_np(boxes).astype(np.float64), to_np(deltas).astype(np.float64)
    center = boxes[:, :3] + from_box_frame(deltas[:, :3], boxes)
    size = boxes[:, 3:6] * np.exp(np.clip(deltas[:, 3:6], -2, 2))   # clip: no overflow from an untrained head
    ry = wrap_angle(boxes[:, 6] + deltas[:, 6])
    return np.concatenate([center, size, ry[:, None]], axis=1).astype(np.float32)


class ResidualDatasetLoader(DatasetLoader):
    '''
    DatasetLoader with proposal-frame inputs and correction targets. Same sampling (Task 2/3 code unchanged);
    every sample also returns its 'proposal' (N,7), needed to decode the network output.
    '''
    def __getitem__(self, idx):
        frame = self.frames[idx]
        points = self.get_data(idx, 'xyz')
        valid_pred, pooled_xyz, pooled_feat = roi_pool(pred=self.get_data(idx, 'detections'),
                                                       xyz=points,
                                                       feat=self.get_data(idx, 'features'),
                                                       config=self.config)
        if self.split == 'test':
            return {'frame': frame, 'proposal': valid_pred.astype(np.float32),
                    'input': np.concatenate((canonicalize(pooled_xyz, valid_pred), pooled_feat), -1)}

        # sample_proposals does not return which proposals it sampled: carry the proposal index as a 4th xyz
        # channel (exact in float32) through the sampling, then split it off
        index = np.broadcast_to(np.arange(len(valid_pred), dtype=pooled_xyz.dtype)[:, None, None],
                                pooled_xyz.shape[:2] + (1,))
        target = self.get_data(idx, 'target')
        assigned_target, xyz, feat, iou = sample_proposals(pred=valid_pred,
                                                           target=target,
                                                           xyz=np.concatenate((pooled_xyz, index), -1),
                                                           feat=pooled_feat,
                                                           config=self.config,
                                                           train=self.split == 'train')
        proposal = valid_pred[xyz[:, 0, 3].astype(int)]
        sampled_frame = {
            'input': np.concatenate((canonicalize(xyz[..., :3], proposal), feat), -1),
            'assigned_target': encode_boxes(proposal, assigned_target),
            'iou': iou,
            'proposal': proposal.astype(np.float32),
        }
        if self.split == 'train':
            return sampled_frame

        sampled_frame.update({
            'frame': frame,
            'target': target,
            'points': points,
        })
        return sampled_frame
