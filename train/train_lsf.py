


import numpy as np

import torch


from utils import AverageMeter
from train.scheduler import warmup_learning_rate


def train_lsf(cfg, model, model2, criterions, optimizer, train_loader, epoch):

    criterion_lc = criterions["lc"]
    criterion_lm = criterions["lm"]
    criterion_sf = criterions["sf"]
    criterion_lwf = criterions["lwf"]

    # model を train モードに変更
    model.train()

    # 継続学習関連の変数
    target_task = cfg.continual.target_task
    cls_per_task = cfg.continual.cls_per_task

    # 学習記録
    losses = AverageMeter()
    accuracies = AverageMeter()    

    corr = [0.] * (cfg.continual.target_task + 1) * cfg.continual.cls_per_task
    cnt  = [0.] * (cfg.continual.target_task + 1) * cfg.continual.cls_per_task
    correct_task = 0.0


    # ニーモニックコードの呼び出し？
    

    for idx, input in enumerate(train_loader):

        images = input["image"]
        labels = input["label"]

        if torch.cuda.is_available():
            # images = images.to(device, non_blocking=True)
            # labels = labels.to(device, non_blocking=True)
            images = images.cuda()
            labels = labels.cuda()

        # バッチサイズの取得
        bsz = labels.shape[0]

        # warm-up
        warmup_learning_rate(cfg, epoch, idx, len(train_loader), optimizer)


        # モデルにデータを入力して出力を取得
        assert False

