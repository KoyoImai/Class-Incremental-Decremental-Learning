


import torch
import torch.nn as nn
import torch.nn.functional as F



class FCSLoss(nn.Module):

    """
    FCS: Feature Calibration and Separation for Non-Exemplar CIL の loss をまとめたクラス。

    このクラスは「テンソルからスカラー loss を計算する」だけに専念します。
    - プロトタイプの構築
    - FCN (Feature Calibration Network) の forward
    - momentun encoder や queue の更新
    などは train 側で行い、ここには既に計算済みの logits / features を渡す設計にしています。

    含まれる loss:
      - L_ce:           通常の cross-entropy
      - L_fkd:          feature distillation (旧モデル feature との L2)
      - L_transfer:     transfer loss (FCN(f_old) と f_new の L2)
      - L_proto:        prototype classification (calibrated prototype を classifier に通した CE)
      - L_contrast:     prototype-involved contrast (PIC) 用の CE
    """


    def __init__(self,
                 lambda_fkd: float=10.0,
                 lambda_proto: float=10,
                 lambda_transfer: float = 1.0,
                 lambda_contrast: float = 0.1,
                 temp_proto: float = 0.1,
                 temp_contrast: float = 0.07,):
        
        super().__init__()
        self.lambda_fkd = lambda_fkd
        self.lambda_proto = lambda_proto
        self.lambda_transfer = lambda_transfer
        self.lambda_contrast = lambda_contrast
        self.temp_proto = temp_proto
        self.temp_contrast = temp_contrast


    @staticmethod
    def l2loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mean: bool = True,
    ) -> torch.Tensor:
        """
        公式 FCS 実装と同じ l2loss の挙動。

        mean = False:
            sqrt(sum ( (x - y)^2 ))   # バッチ全体をまとめて1スカラー
        mean = True:
            sqrt(sum_d ( (x - y)^2 )) をサンプル毎に計算 → 平均
        """
        delta = (inputs - targets) ** 2

        if mean:
            # (B, D) -> (B,) -> scalar
            delta = torch.sqrt(delta.sum(dim=-1))
            return delta.mean()
        else:
            # 全要素をまとめて L2
            return torch.sqrt(delta.sum())








