
import torch
import torch.nn as nn
import torch.nn.functional as F



from utils import seed_everything
from models.resnet_cifar import model_dict  # ResNet本体と出力次元の対応を再利用




class BackboneResNet(nn.Module):

    def __init__(self, name: str="resnet18", head: str="linear", cfg=None):

        super().__init__()

        # 継続学習の設定
        self.num_task = cfg.continual.num_task        # タスク数
        self.cls_per_task = cfg.continual.cls_per_task

        # 再現性
        seed_everything(seed=cfg.seed)

        # encoder の構築
        model_fun, dim_in = model_dict[name]          # resnet18 -> (constructor, feature_dim)
        self.encoder = model_fun()
        self.feature_dim = dim_in                     # FCS Loss 側で使いやすいように持っておく

        # 出力層の output サイズ
        output_size = cfg.continual.output_size

        if head == "linear":
            layers = [nn.Linear(dim_in, output_size[i], bias=True) for i in range(self.num_task)]
            self.head = nn.ModuleList(layers)

        else:
            assert False

        # ==== Feature Calibration Network (FCN) ====
        # f_old -> f_new を近似する線形写像
        self.transfer = nn.Linear(dim_in, dim_in, bias=True)

    
    def forward(self, x, taskid=None):

        """
        x: (B, C, H, W)
        taskid: 1〜(num_task) の整数（None のときは全タスクまで）
        return:
            logits: (B, sum_{t < taskid} output_size[t])
            encoded: (B, feature_dim)  ← conv feature（FCS Loss で使う）
        """
        # 出力を取り出すタスク範囲の決定
        if taskid is None:
            taskid = self.num_task
        
        # backbone に画像を入力して特徴抽出
        encoded = self.encoder(x)   # (B, feature_dim)

        # head に特徴を入力して各タスクの logits を作る
        logits_list = []
        for i in range(taskid):
            outputs = self.head[i](encoded)   # (B, output_size[i])
            logits_list.append(outputs)

        logits = torch.cat(logits_list, dim=1) if len(logits_list) > 0 else None

        return logits, encoded


    @torch.no_grad()
    def extract_feature(self, x):
        """
        旧モデルの特徴抽出などで使いたい場合用のユーティリティ。
        """
        self.encoder.eval()
        return self.encoder(x)
    
    
    def calibrate_features(self, feats):
        """
        FCN を通して特徴を補正するための helper。
        feats: (..., feature_dim)
        """
        return self.fcn(feats)
    

    def print_model(self):
        print(self.encoder)
        print(self.head)
        print(self.transfer)






