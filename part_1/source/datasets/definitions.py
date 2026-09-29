MOD_ID = 'id'
MOD_RGB = 'rgb'
MOD_SEMSEG = 'semseg'
MOD_DEPTH = 'depth'

SPLIT_TRAIN = 'train'
SPLIT_VALID = 'val'
SPLIT_TEST = 'test'

INVALID_VALUE = float('-inf')

INTERP = {
    MOD_ID: None,
    MOD_RGB: 'bilinear',
    MOD_SEMSEG: 'nearest',
    MOD_DEPTH: 'nearest',
}
