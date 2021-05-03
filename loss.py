import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input_tensor, target_tensor):
        return F.cross_entropy(input_tensor, target_tensor)


class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=1., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=21, smoothing=0.2, dim=1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=21, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, smooth=1):
        """
            Args:
                inputs: (N, C, W, H)
                targets: (N, W, H)
        """ 
        num_classes = inputs.shape[1]

        true_1_hot = torch.eye(num_classes)[targets.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(inputs, dim=1)

        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()

        return (1 - dice_loss)          # in order to maximize dice loss


class CEandDICE(nn.Module):
    def __init__(self, coeff=1):
        super(CEandDICE, self).__init__()

        self.CE = CrossEntropyLoss()
        self.DICE = DiceLoss()
        self.coeff = coeff

    def forward(self, inputs, targets):
        return self.CE(inputs, targets) + self.coeff * self.DICE(inputs, targets) 


class CEandLabelSmoothing(nn.Module):
    def __init__(self, coeff=1):
        super(CEandLabelSmoothing, self).__init__()

        self.CE = CrossEntropyLoss()
        self.LabelSmoothing = LabelSmoothingLoss()
        self.coeff = coeff

    def forward(self, inputs, targets):
        return self.CE(inputs, targets) + self.coeff * self.LabelSmoothing(inputs, targets)