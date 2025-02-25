import torch
import torch.nn as nn
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from wilds.common.metrics.all_metrics import MSE
from utils import cross_entropy_with_logits_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return torch.ones_like(y)
        return torch.where(y.bool(), self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = torch.clip(p, 1e-15, 1 - 1e-15)
        return torch.where(y.bool(), p, 1 - p)

    def forward(self, y_pred, y_true):
        y_pred = torch.argmax(y_pred, dim=1)
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * torch.log(pt)


def initialize_loss(loss, config):
    if loss == 'cross_entropy':
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    if loss == "focal_cross_entropy":
        return ElementwiseLoss(loss_fn=FocalLoss(gamma=2.0, alpha=None))

    elif loss == 'lm_cross_entropy':
        return MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    elif loss == 'mse':
        return MSE(name='loss')

    elif loss == 'multitask_bce':
        return MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))

    elif loss == 'fasterrcnn_criterion':
        from models.detection.fasterrcnn import FasterRCNNLoss
        return ElementwiseLoss(loss_fn=FasterRCNNLoss(config.device))

    elif loss == 'cross_entropy_logits':
        return ElementwiseLoss(loss_fn=cross_entropy_with_logits_loss)

    else:
        raise ValueError(f'loss {loss} not recognized')
