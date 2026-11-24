import torch
import torch.nn as nn

class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        positive = iou >= self.config['positive_reg_lb']
        if not positive.any():
            return pred.sum() * 0.0          # no positives in the batch: zero loss, but keep the graph for backward()
        pred, target = pred[positive], target[positive]
        loss_location = self.loss(pred[:, 0:3], target[:, 0:3])
        loss_size = self.loss(pred[:, 3:6], target[:, 3:6])
        loss_rotation = self.loss(pred[:, 6], target[:, 6])
        return loss_location + 3 * loss_size + loss_rotation

class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,7) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        pred = pred.reshape(-1)                          # (N,1) sigmoid scores from the model -> (N,)
        positive = iou >= self.config['positive_cls_lb']
        negative = iou <= self.config['negative_cls_ub']
        used = positive | negative                       # proposals in between give no training signal
        if not used.any():
            return pred.sum() * 0.0
        return self.loss(pred[used], positive[used].to(pred.dtype))
