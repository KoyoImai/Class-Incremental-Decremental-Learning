

import os
import numpy as np


import torch


from utils import AverageMeter


def train_finetune(cfg, model, model2, criterions, optimizer, train_loader, epoch):

    criterion = criterions["ce"]
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

        # モデルにデータを入力して出力を取得
        y_pred, _ = model(images, taskid=target_task+1)
        # print("y_pred.shape: ", y_pred.shape)

        # 損失を計算
        loss = criterion(y_pred, labels).mean()

        # update metric
        losses.update(loss.item(), bsz)


        # 正解率の計算
        cls_list = np.unique(labels.cpu())
        correct_all = (y_pred.argmax(1) == labels)

        for tc in cls_list:
            mask = labels == tc
            correct_task += (y_pred[mask, (tc // cls_per_task) * cls_per_task : ((tc // cls_per_task)+1) * cls_per_task].argmax(1) == (tc % cls_per_task)).float().sum()

        for c in cls_list:
            mask = labels == c
            corr[c] += correct_all[mask].float().sum().item()
            cnt[c] += mask.float().sum().item()

        if idx % 10 == 0:
            print('Train: [{0}/{1}]\t'
                    'Acc@1 {top1:.3f} {task_il:.3f}'.format(
                        idx, len(train_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.
                    ))


        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()






