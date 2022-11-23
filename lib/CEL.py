from torch import nn


class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)
