

import os
import numpy as np

import torch


from utils import AverageMeter
from train.scheduler import warmup_learning_rate



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

        # warm-up
        warmup_learning_rate(cfg, epoch, idx, len(train_loader), optimizer)

        # モデルにデータを入力して出力を取得
        y_pred, _ = model(images, taskid=target_task+1)
        # print("y_pred.shape: ", y_pred.shape)

        # 損失を計算
        loss = criterion(y_pred, labels).mean()

        # update metric
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




def print_class_wise_acc(corr, cnt, num_classes):

    corr = np.array(corr)
    cnt  = np.array(cnt)
    per_class_acc = np.zeros_like(corr)
    valid = cnt > 0
    per_class_acc[valid] = corr[valid] / cnt[valid] * 100.

    print("\n=== Per-class accuracy (Class-IL) ===")
    for cls_id in range(num_classes):
        if cnt[cls_id] == 0:
            continue  # そのクラスが一度も出てきてない
        print(f"Class {cls_id:3d}: {per_class_acc[cls_id]:6.2f}%  (n={int(cnt[cls_id])})")





def eval_finetune(cfg, model, model2, criterions, optimizer, val_loader, epoch):

    criterion = criterions["ce"]

    # model を eval モードに変更
    model.eval()

    # 継続学習関連の変数
    target_task = cfg.continual.target_task
    cls_per_task = cfg.continual.cls_per_task

    # 全クラス数（タスク 0〜target_task まで）
    num_classes = (target_task + 1) * cls_per_task

    # タスク毎の精度を保持
    corr = [0.] * num_classes
    cnt  = [0.] * num_classes
    correct_task = 0.0

    losses = AverageMeter()

    with torch.no_grad():

        for idx, input in enumerate(val_loader):

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

            losses.update(loss.item(), bsz)

            cls_list = np.unique(labels.cpu())
            correct_all = (y_pred.argmax(1) == labels)

            for tc in cls_list:
                mask = labels == tc
                correct_task += (y_pred[mask, (tc // cls_per_task) * cls_per_task : ((tc // cls_per_task)+1) * cls_per_task].argmax(1) == (tc % cls_per_task)).float().sum()

            for c in cls_list:
                mask = labels == c
                corr[c] += correct_all[mask].float().sum().item()
                cnt[c] += mask.float().sum().item()

            if idx % 1 == 0:
                print('Test: [{0}/{1}]\t'
                      'Acc@1 {top1:.3f} {task_il:.3f}'.format(
                          idx, len(val_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.
                      ))
    
    top1=np.sum(corr)/np.sum(cnt)*100.
    task_il=correct_task/np.sum(cnt)*100.
    print(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=top1, task_il=task_il))


    # クラス毎の精度表示
    print_class_wise_acc(corr, cnt, num_classes)

    return top1, task_il




