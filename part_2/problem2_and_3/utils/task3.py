import numpy as np

from .task1 import get_iou
from .task2 import sample_indices


def assign_targets(pred, target):
    '''
    Match every proposal with the ground-truth box it overlaps most.
    input
        pred (N,7) proposals, target (M,7) ground truth
    output
        assigned (N,7) best-matching target of each proposal
        best_iou (N,) IoU with that target
        iou (N,M) full IoU matrix
    '''
    iou = get_iou(pred, target)
    best = iou.argmax(axis=1)
    return target[best], iou[np.arange(len(pred)), best], iou


def split_proposals(best_iou, iou, config):
    '''
    Indices of the foreground, easy background and hard background proposals.
    Foreground: IoU >= t_fg_lb, plus, for every ground-truth box, the proposal with the highest
    IoU (if > 0) as an additional, repeated sample. Proposals with t_bg_up <= IoU < t_fg_lb are
    ignored unless they are added by that rule.
    '''
    fg = np.flatnonzero(best_iou >= config['t_fg_lb'])
    best_for_target = iou.argmax(axis=0)[iou.max(axis=0) > 0]
    fg = np.concatenate([fg, best_for_target])
    easy_bg = np.flatnonzero(best_iou < config['t_bg_hard_lb'])
    hard_bg = np.flatnonzero((best_iou >= config['t_bg_hard_lb']) & (best_iou < config['t_bg_up']))
    return fg, easy_bg, hard_bg


def sample_background(easy_bg, hard_bg, n, hard_ratio):
    '''n background indices: hard_ratio of them hard (rounded down, so an odd extra comes from the easy pool)
    when both pools exist, otherwise all from the pool that exists.'''
    if n == 0:
        return np.zeros(0, dtype=int)
    if len(easy_bg) and len(hard_bg):
        n_hard = int(n * hard_ratio)
        return np.concatenate([sample_indices(easy_bg, n - n_hard), sample_indices(hard_bg, n_hard)])
    return sample_indices(easy_bg if len(easy_bg) else hard_bg, n)


def sample_proposals(pred, target, xyz, feat, config, train=False):
    '''
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (bool) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices
        feat (64,512,C) indices
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples)
    '''
    assigned, best_iou, iou = assign_targets(pred, target)
    if not train:                                   # validation: every proposal, no sampling
        return assigned, xyz, feat, best_iou

    num_samples = config.get('num_samples', 64)
    fg, easy_bg, hard_bg = split_proposals(best_iou, iou, config)
    has_bg = len(easy_bg) + len(hard_bg) > 0

    if len(fg) and has_bg:
        n_fg = min(len(fg), config['num_fg_sample'])
    elif len(fg):
        n_fg = num_samples                          # only foreground in the scene
    elif has_bg:
        n_fg = 0                                    # only background in the scene
    else:                                           # every proposal in the ignored IoU band: no class to sample from
        idx = sample_indices(np.arange(len(pred)), num_samples)
        return assigned[idx], xyz[idx], feat[idx], best_iou[idx]

    fg_idx = sample_indices(fg, n_fg) if n_fg else np.zeros(0, dtype=int)
    bg_idx = sample_background(easy_bg, hard_bg, num_samples - n_fg, config['bg_hard_ratio'])
    idx = np.concatenate([fg_idx, bg_idx])
    return assigned[idx], xyz[idx], feat[idx], best_iou[idx]
