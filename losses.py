import torch
import torch.nn as nn
import torch.nn.functional as F

"""
focal loss 是一种处理样本分类不均衡的损失函数
Lsum = a1 x L易区分 + a2 x L难区分
因为α1小而α2大，那么上述的损失函数中L难区分L难区分主导损失函数，
也就是将损失函数的重点集中于难分辨的样本上，对应损失函数的名称：focal loss
"""


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target):
        logpt = - F.cross_entropy(output, target)
        pt = torch.exp(logpt)
        focal_loss = -self.alpha * (1-pt) ** self.gamma * logpt
        return focal_loss


class FocalLoss1(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    # criterion = FocalLoss(gamma=2, alpha=torch.tensor([1.0, 2.0, 3.0])) # 三个类别的权重分别为1.0，2.0，3.
    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt)**self.gamma * CE_loss

        if self.alpha is not None:
            assert len(self.alpha) == inputs.shape[1]
            self.alpha = self.alpha.to(inputs.device)
            F_loss = self.alpha[targets] * F_loss

        return torch.mean(F_loss)
