
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import seed_everything
from models.resnet_cifar import model_dict, ResNet, Bottleneck, BasicBlock



# def resnet18(**kwargs):
#     return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


# def resnet34(**kwargs):
#     return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


# def resnet50(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def resnet101(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


# model_dict = {
#     'resnet18': [resnet18, 512],
#     'resnet34': [resnet34, 512],
#     'resnet50': [resnet50, 2048],
#     'resnet101': [resnet101, 2048]
# }


class BackboneResNet(nn.Module):

    def __init__(self, name="resnet18", head="linear", cfg=None):

        super(BackboneResNet, self).__init__()

        # 変数関連の初期化
        self.num_task = cfg.continual.num_task
        self.cls_per_task = cfg.continual.cls_per_task

        # seed値の設定
        seed_everything(seed=cfg.seed)

        # モデルの初期化
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()

        # 出力層の output サイズ
        output_size = cfg.continual.output_size

        if head == "linear":
            layers = [nn.Linear(dim_in, output_size[i], bias=True) for i in range(self.num_task)]
            self.head = nn.ModuleList(layers)

        else:
            assert False


    # def forward(self, x, taskid=None):

    #     # 出力を取り出す範囲の決定
    #     if taskid is None:
    #         taskid = self.num_task

    #     # backboneに画像を入力
    #     encoded = self.encoder(x)

    #     # head に backbone の出力を入力
    #     logits_list = []
    #     for i in range(taskid):
    #         outputs = self.head[i](encoded)
    #         # print("outputs.shape: ", outputs.shape)    # outputs.shape:  torch.Size([128, 2])
    #         logits_list.append(outputs)
        
    #     logits = torch.cat(logits_list, dim=1)      

    #     return logits, encoded
    

    def forward(self, x, taskid=None):

        # 出力を取り出す範囲の決定
        if taskid is None:
            taskid = self.num_task

        # backboneに画像を入力
        encoded = self.encoder(x)  # (B, dim_in)

        # --- ここから公式 LsF 風に調整 ---

        # 特徴を L2 正規化して cosθ 用ベクトルにする
        feat = F.normalize(encoded, p=2, dim=1)  # (B, dim_in)

        logits_list = []
        for i in range(taskid):
            # head の weight を L2 正規化
            w = self.head[i].weight  # (num_cls_task_i, dim_in)
            w_norm = F.normalize(w, p=2, dim=1)

            # bias は使わず F.linear で cosθ を計算
            logits_i = F.linear(feat, w_norm)  # (B, num_cls_task_i)
            logits_list.append(logits_i)

        logits = torch.cat(logits_list, dim=1)  # (B, sum_{t<=taskid} cls_per_task)

        return logits, encoded



    def print_model(self):

        print(self.encoder)
        print(self.head)

