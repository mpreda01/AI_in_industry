import numpy as np

from utils.task1 import get_iou


def bev_iou(boxes_a, boxes_b):
    '''
    Bird's-eye-view IoU of (N,7) and (M,7) boxes (x,y,z,h,w,l,ry): the overlap of their x-z footprints.
    Reuses the 3D IoU of task 1 with every box flattened to the same vertical range (y, h) = (0, 1),
    so the height overlap is always 1 and only the footprint counts.
    '''
    def flatten(boxes):
        boxes = np.array(boxes, dtype=np.float64).reshape(-1, 7)
        boxes[:, 1], boxes[:, 3] = 0.0, 1.0
        return boxes
    return get_iou(flatten(boxes_a), flatten(boxes_b))


def nms(pred, score, threshold):
    '''
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresopnding confidence scores
    '''
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 7)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    iou = bev_iou(pred, pred)

    keep = []
    removed = np.zeros(len(pred), dtype=bool)
    for i in np.argsort(-score, kind='stable'):      # most confident first; ties keep their original order
        if removed[i]:
            continue
        keep.append(i)
        removed |= iou[i] >= threshold                # drop everything that overlaps the kept box too much
    return pred[keep], score[keep].reshape(-1, 1)
