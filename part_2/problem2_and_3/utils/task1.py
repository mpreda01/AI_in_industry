import numpy as np

# Corner order (same as problem1/3dvis.py):
#     1 -------- 0          top face    (y = y_c - h)
#    /|         /|
#   2 -------- 3 .
#   | |        | |
#   . 5 -------- 4          bottom face (y = y_c)
#   |/         |/
#   6 -------- 7
# Length runs along the box's local x axis, width along its local z axis; 4-5 is the front edge.
_CORNER_X = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * 0.5
_CORNER_Y = np.array([-1, -1, -1, -1, 0, 0, 0, 0])
_CORNER_Z = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * 0.5

_EPS = 1e-9
_MAX_PAIRS_PER_CHUNK = 100_000


def label2corners(label):
    '''
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
    '''
    label = np.asarray(label, dtype=np.float64).reshape(-1, 7)
    x, y, z, h, w, l, ry = (label[:, k:k + 1] for k in range(7))

    # Corners in the box's own frame: (x, y, z) is the center of the bottom face, y points down
    xs, ys, zs = _CORNER_X * l, _CORNER_Y * h, _CORNER_Z * w

    # Rotate around the vertical (y) axis by ry, then translate
    c, s = np.cos(ry), np.sin(ry)
    return np.stack([c * xs + s * zs + x,
                     ys + y,
                     -s * xs + c * zs + z], axis=-1)


def _cross(a, b):
    '''z-component of the 2D cross product, broadcast over leading axes.'''
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inside(points, poly):
    '''points (P,K,2), poly (P,4,2) counter-clockwise convex -> (P,K) True if the point is inside or on the border.'''
    edges = np.roll(poly, -1, axis=1) - poly                        # (P,4,2)
    rel = points[:, :, None, :] - poly[:, None, :, :]               # (P,K,4,2)
    return (_cross(edges[:, None], rel) >= -_EPS).all(axis=-1)


def _intersection_area(a, b):
    '''
    Area of the intersection of pairs of convex quadrilaterals.
    a, b (P,4,2) corners in counter-clockwise order -> (P,) areas

    The intersection of two convex polygons is a convex polygon whose vertices are
    (1) corners of a inside b, (2) corners of b inside a and (3) crossings of an edge
    of a with an edge of b. Collect all of them (at most 4 + 4 + 16 = 24), sort them by
    angle around their centroid and apply the shoelace formula.
    '''
    P = len(a)
    ra, rb = np.roll(a, -1, axis=1) - a, np.roll(b, -1, axis=1) - b      # edge vectors

    # Edge crossings: a_i + t * ra_i = b_j + u * rb_j with t, u in [0, 1]
    p, r = a[:, :, None, :], ra[:, :, None, :]                          # (P,4,1,2)
    q, s = b[:, None, :, :], rb[:, None, :, :]                          # (P,1,4,2)
    denom = _cross(r, s)                                                # (P,4,4)
    parallel = np.abs(denom) < _EPS
    denom = np.where(parallel, 1.0, denom)
    t = _cross(q - p, s) / denom
    u = _cross(q - p, r) / denom
    cross_ok = ~parallel & (t >= -_EPS) & (t <= 1 + _EPS) & (u >= -_EPS) & (u <= 1 + _EPS)
    cross_pts = (p + t[..., None] * r).reshape(P, 16, 2)

    pts = np.concatenate([a, b, cross_pts], axis=1)                      # (P,24,2)
    valid = np.concatenate([_inside(a, b), _inside(b, a), cross_ok.reshape(P, 16)], axis=1)
    count = valid.sum(axis=1)

    # Sort valid vertices by angle around their centroid; invalid ones go to the end
    centroid = (pts * valid[..., None]).sum(axis=1) / np.maximum(count, 1)[:, None]
    angle = np.arctan2(pts[..., 1] - centroid[:, None, 1], pts[..., 0] - centroid[:, None, 0])
    angle = np.where(valid, angle, np.inf)
    pts = np.take_along_axis(pts, np.argsort(angle, axis=1)[..., None], axis=1)

    # Replace the invalid tail by copies of the last valid vertex: zero-length edges add no area
    idx = np.minimum(np.arange(24)[None, :], np.maximum(count, 1)[:, None] - 1)
    pts = np.take_along_axis(pts, idx[..., None], axis=1)

    area = 0.5 * np.abs(_cross(pts, np.roll(pts, -1, axis=1)).sum(axis=1))
    return np.where(count >= 3, area, 0.0)


def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union

    Both inputs are box labels (x,y,z,h,w,l,ry) in the rectified camera frame.
    The 3D intersection is the bird's-eye-view (x-z) overlap times the vertical (y) overlap,
    since all boxes only rotate around the vertical axis.
    '''
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 7)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 7)
    iou = np.zeros((len(pred), len(target)))
    if len(pred) == 0 or len(target) == 0:
        return iou

    # Vertical overlap: y points down, so a box spans [y - h, y]
    top = np.maximum(pred[:, None, 1] - pred[:, None, 3], target[None, :, 1] - target[None, :, 3])
    bottom = np.minimum(pred[:, None, 1], target[None, :, 1])
    h_overlap = np.clip(bottom - top, 0.0, None)

    # Only pairs whose bird's-eye-view circumscribed circles touch can overlap.
    # Boxes with non-positive sizes (e.g. the dummy box get_data returns for frames without cars) never overlap.
    radius_p, radius_t = 0.5 * np.hypot(pred[:, 4], pred[:, 5]), 0.5 * np.hypot(target[:, 4], target[:, 5])
    center_dist = np.hypot(pred[:, None, 0] - target[None, :, 0], pred[:, None, 2] - target[None, :, 2])
    valid_p, valid_t = (pred[:, 3:6] > 0).all(axis=1), (target[:, 3:6] > 0).all(axis=1)
    candidate = ((h_overlap > 0) & (center_dist < radius_p[:, None] + radius_t[None, :])
                 & valid_p[:, None] & valid_t[None, :])
    i, j = np.nonzero(candidate)
    if len(i) == 0:
        return iou

    bev_p = label2corners(pred)[:, 4:8][:, :, [0, 2]]       # bottom face in the x-z plane
    bev_t = label2corners(target)[:, 4:8][:, :, [0, 2]]
    vol_p = pred[:, 3:6].prod(axis=1)
    vol_t = target[:, 3:6].prod(axis=1)
    for start in range(0, len(i), _MAX_PAIRS_PER_CHUNK):     # bounded memory for large inputs (eval.py)
        ci, cj = i[start:start + _MAX_PAIRS_PER_CHUNK], j[start:start + _MAX_PAIRS_PER_CHUNK]
        inter = _intersection_area(bev_p[ci], bev_t[cj]) * h_overlap[ci, cj]
        iou[ci, cj] = inter / (vol_p[ci] + vol_t[cj] - inter)
    return iou


def compute_recall(pred, target, threshold):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene

    A ground-truth box counts as found (true positive) if at least one proposal has
    IoU >= threshold with it; otherwise it's a false negative. Recall = TP / (TP + FN).
    Scenes without ground truth have an undefined recall and return nan (use np.nanmean to average).
    '''
    target = np.asarray(target, dtype=np.float64).reshape(-1, 7)
    target = target[(target[:, 3:6] > 0).all(axis=1)]       # drop get_data's dummy box for frames without cars
    if len(target) == 0:
        return np.nan
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 7)
    if len(pred) == 0:
        return 0.0
    found = get_iou(pred, target).max(axis=0) >= threshold
    return float(found.mean())
