

import os

import torch
import torch.nn as nn
import torch.nn.functional as F



class AdditiveMarginSoftmaxLoss(nn.Module):
    """
    通常の交差エントロピー損失のラッパー（他の損失で使用します）．
    """

    def __init__(self,
                 scale: float=30,
                 margin: float=0.35,
                 reduction: str="mean"):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.reduction = reduction
    

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)
        targets: (B,)
        """

        num_classes = logits.size(1)
        one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # 正解クラスだけ margin を引く
        logits_m = logits - one_hot * self.margin
        logits_s = logits_m * self.scale  # 全クラスに scale

        loss = F.cross_entropy(logits_s, targets, reduction=self.reduction)
        return loss


class LCLoss(nn.Module):
    """
    LSFの分類損失．
    """

    def __init__(self,
                 ams_scale: float=30,
                 ams_margin: float=0.35,
                 reduction: str="mean",
                 weight: float=1.0):
        
        super().__init__()
        self.weight = weight
        self.ams = AdditiveMarginSoftmaxLoss(
            scale=ams_scale,
            margin=ams_margin,
            reduction="mean",
        )
    

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)  → cos(theta)
        targets: (B,)
        """
        return self.weight * self.ams(logits, targets)
    

class LMLoss(nn.Module):
    """
    LSFのニーモニックコードを埋め込んだサンプルによる分類損失．
    """
    def __init__(self,
                 ams_scale: float=30,
                 ams_margin: float=0.35,
                 reduction: str="mean",
                 weight: float=1.0):
        
        super().__init__()
        self.weight = weight
        self.ams = AdditiveMarginSoftmaxLoss(
            scale=ams_scale,
            margin=ams_margin,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)
        targets: (B,)
        """
        return self.weight * self.ams(logits, targets)



class SelectiveForgettingLoss(nn.Module):
    """
    LSFのニーモニックコードを対象とした忘却損失．
    """
    def __init__(self,
                 ams_scale: float = 30.0,
                 ams_margin: float = 0.35,
                 gamma_sf: float = 1.0,):
        
        super().__init__()
        self.gamma_sf = gamma_sf
        self.ams = AdditiveMarginSoftmaxLoss(
            scale=ams_scale,
            margin=ams_margin,
            reduction="mean",
        )

    def forward(self):

        assert False




class LwFLoss(nn.Module):

    def __init__(self, gamma_lwf: float=1.0, T: float=2.0):
        super().__init__()
        self.gamma_lwf = gamma_lwf
        self.T = T

    def forward(self):

        assert False
        

