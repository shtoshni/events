import torch.nn as nn
import torch


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target, weight=None):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.shape[self.dim] - 1))
            true_dist.scatter_(0, target, self.confidence)

        if weight is None:
            loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        else:
            loss = torch.sum(-true_dist * pred * weight, dim=self.dim) / torch.sum(weight)

        return loss


class LabelSmoothingLossOther(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossOther, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target, weight=None):
        pred = pred.log_softmax(dim=-1)

        true_dist = torch.nn.functional.softmax(pred.detach(), dim=0)
        target_prob = true_dist.gather(dim=0, index=target)

        true_dist = (target_prob >= self.confidence) * true_dist + \
                    (target_prob < self.confidence) * torch.ones_like(true_dist)
        # First set the true distribution to be 0
        true_dist.scatter_(0, target, 0)
        # Now distribute the smoothing probability mass among other labels in proportion to predicted probs
        true_dist = (true_dist * self.smoothing) / (torch.sum(true_dist) + 1e-8)
        # Now set the true label with (1 - smoothing) probability
        true_dist.scatter_(0, target, self.confidence)
        loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        return loss