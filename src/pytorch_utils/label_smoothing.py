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

        with torch.no_grad():
            # true_dist = pred.softmax(dim=self.dim)
            # # First set the true distribution to be 0
            # true_dist.scatter_(self.dim, target, 0)
            # # Now distribute the smoothing probability mass among other labels in proportion to predicted probs
            # true_dist = (true_dist * self.smoothing) / (torch.sum(true_dist, dim=self.dim) + 1e-8)
            # # Now set the true label with (1 - smoothing) probability
            # true_dist.scatter_(self.dim, target, self.confidence)
            # print(true_dist)
            # # true_dist = pred.data.clone()
            # true_dist = torch.zeros_like(pred)
            # true_dist.fill_(self.smoothing / (pred.shape[self.dim] - 1))
            # true_dist.scatter_(self.dim, target, self.confidence)

            soft_pred = pred.softmax(dim=0)
            # target_prob = soft_pred.gather(dim=0, index=target).squeeze(dim=0)
            max_prob, argmax_labels = soft_pred.max(dim=-1)

            soft_pred.scatter_(0, argmax_labels, 0)
            other_max_prob = soft_pred.max(dim=-1)[0]

        nll_loss = torch.nn.functional.cross_entropy(torch.unsqueeze(pred, dim=0), target)
        max_margin_loss = torch.max(torch.ones_like(max_prob) * self.confidence, max_prob - other_max_prob)

        loss = nll_loss + max_margin_loss

        return loss