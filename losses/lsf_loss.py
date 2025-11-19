

import os

import torch
import torch.nn as nn
import torch.nn.functional as F



def cmp_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    公式実装の cmp_entropy をそのまま再現。

        f = softmax(logits)
        f = log(f + 1e-10)
        return -sum(f)

    ※標準的な -sum(p log p) ではなく、-sum(log p) になっている点に注意。
    """
    f = F.softmax(logits, dim=1)
    f = (f + 1e-10).log()
    return -f.sum()



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
    

    def forward(self, logits: torch.Tensor, targets, weight: float = 1.0) -> torch.Tensor:
        """
        Args:
            logits: (B, C)  cosθ
            targets: (B,) int もしくは (B, C) one-hot
            weight: 全体に掛ける係数
        """
        # targets をクラス index に揃える
        if targets.dim() == 1:
            y = targets
        else:
            # one-hot → index
            y = targets.argmax(dim=1)

        logits = logits.clone()

        # 正解クラスに margin を引く
        # ここで CE が内部で log-sum-exp を使うので数値的に安定
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        logits[batch_indices, y] = logits[batch_indices, y] - self.margin

        # scale を掛けて普通の CrossEntropy
        logits = logits * self.scale
        loss = F.cross_entropy(logits, y)

        return weight * loss



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
        return self.ams(logits, targets, weight=self.weight)
    

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

    def forward(self,
                logits_keep: torch.Tensor,
                targets_keep: torch.Tensor,
                weight_keep: float = 1.0) -> torch.Tensor:
        """
        Args:
            logits_keep:  preservation set に属する mnemonic 入力の logits (B, C)
            targets_keep: そのクラスラベル (B,)
            weight_keep:  AMS に渡す係数（通常は1.0）

        Returns:
            γ_SF * ( AMS(logits_keep, targets_keep) )
        """
        if logits_keep is None or targets_keep is None:
            # ここに来るのは target_task == 0 のときなどだけのはず
            device = None
            if isinstance(logits_keep, torch.Tensor):
                device = logits_keep.device
            elif isinstance(targets_keep, torch.Tensor):
                device = targets_keep.device
            return torch.tensor(0.0, device=device)

        loss_keep = self.ams(logits_keep, targets_keep, weight=weight_keep)
        return self.gamma_sf * loss_keep



class LwFLoss(nn.Module):
    """
    LwF* の distillation loss.

    論文中の式:
        L_LwF* = - γ_lwf * Σ_{i ∈ C^P_k} y'_o(i) log ŷ'_o(i)

    をミニバッチ単位で実装したもの。
    - teacher_logits: 古いモデル（model2）のロジット
    - student_logits: 現在のモデル（model）のロジット
    - keep_indices: preservation set C^P_k に属するクラス index の 1D LongTensor
                    ※ None なら全クラスを対象に distillation
    """

    def __init__(self, gamma_lwf: float = 1.0, T: float = 2.0):
        super().__init__()
        self.gamma_lwf = gamma_lwf
        self.T = T

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        keep_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: (B, C)  現在のモデルのロジット
            teacher_logits: (B, C)  古いモデルのロジット（勾配は流さない）
            keep_indices:  (M,)     distillation に使うクラス index のリスト
                                  （preservation set C^P_k に対応）

        Returns:
            LwF* のスカラー損失
        """
        device = student_logits.device
        T = self.T

        # preservation set のクラスだけを抜き出す
        if keep_indices is not None:
            if keep_indices.numel() == 0:
                # 保持すべきクラスが無い（全部忘却対象）の場合は 0
                return torch.tensor(0.0, device=device)

            # (B, M)
            student_logits = student_logits[:, keep_indices]
            teacher_logits = teacher_logits[:, keep_indices]

        # distillation 用の分布
        # y'_o, ŷ'_o に対応（softmax(logits / T) は論文の定義と等価）
        log_p = F.log_softmax(student_logits / T, dim=1)  # 学習側
        with torch.no_grad():
            q = F.softmax(teacher_logits / T, dim=1)       # 教師側

        # KL(p || q) * T^2 （Hinton 論文＋公式実装と同じスケーリング）
        loss_kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)

        return self.gamma_lwf * loss_kl

