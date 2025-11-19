


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


    # ===========================================
    # ニーモニックコード関連の準備（まだ未実装）
    # ===========================================
    mnemonic_buffer = None  # TODO: 後で実装

    
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
        y_pred, _ = model(images, taskid=target_task+1)

        # 通常の交差エントロピー損失を計算
        loss_lc = criterion_lc(y_pred, labels)

        # ニーモニックコードをmixupしたサンプルによる損失
        loss_lm = torch.zeros(1, device=images.device)

        # 選択的忘却損失の計算
        loss_sf = torch.zeros(1, device=images.device)

        # 蒸留損失の計算
        loss_lwf = torch.zeros(1, device=images.device)

        # 損失を合計
        loss = loss_lc + loss_lm + loss_sf + loss_lwf

        losses.update(loss.item(), bsz)


        # 正解率の計算
        with torch.no_grad():
            cls_list = np.unique(labels.cpu())
            correct_all = (y_pred.argmax(1) == labels)


            for c in cls_list:
                mask = labels == c
                corr[c] += correct_all[mask].float().sum().item()
                cnt[c] += mask.float().sum().item()

        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']

        if idx % 10 == 0:
            print('Train\t' 
                  'Epoch: {0:2d} [{1}/{2}]\t'
                  'LR: {lr:.5f}\t'
                  'Loss: {loss:.5f}\t'
                  'Acc@1 {top1:.3f}'.format(epoch, idx, len(train_loader), lr=current_lr, loss=losses.avg, top1=np.sum(corr)/np.sum(cnt)*100.))


        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()