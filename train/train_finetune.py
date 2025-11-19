

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
                  'Epoch: [{0:2d}/{1:2d}] [{2}/{3}]\t'
                  'LR: {lr:.5f}\t'
                  'Loss: {loss:.5f}\t'
                  'Acc@1 {top1:.3f}'.format(epoch, cfg.optimizer.train.epoch, idx, len(train_loader), lr=current_lr, loss=losses.avg, top1=np.sum(corr)/np.sum(cnt)*100.))


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

    print("=== Per-class accuracy (Class-IL) ===")
    for cls_id in range(num_classes):
        if cnt[cls_id] == 0:
            continue  # そのクラスが一度も出てきてない
        print(f"Class {cls_id:3d}: {per_class_acc[cls_id]:6.2f}%  (n={int(cnt[cls_id])})")



def print_task_wise_acc(task_corr, task_cnt, num_tasks):
    print("=== Task-wise accuracy (Task-IL) ===")
    for t in range(num_tasks):
        if task_cnt[t] == 0:
            acc_t = 0.0
        else:
            acc_t = task_corr[t] / task_cnt[t] * 100.0
        print(f"  Task {t}: Acc {acc_t:.3f}% ({int(task_corr[t])}/{int(task_cnt[t])})")


def print_forget_keep_acc(forget_cnt_ci, forget_corr_ci, keep_cnt_ci, keep_corr_ci):
    print('=== Forget-Acc & Keep-Acc (Class-IL)  ===')
    if forget_cnt_ci > 0:
        forget_acc_ci = forget_corr_ci / forget_cnt_ci * 100.0
    else:
        forget_acc_ci = 0.0

    if keep_cnt_ci > 0:
        keep_acc_ci = keep_corr_ci / keep_cnt_ci * 100.0
    else:
        keep_acc_ci = 0.0

    # 調和平均（両方0なら0）
    if (forget_acc_ci + keep_acc_ci) > 0:
        hm_ci = 2.0 * forget_acc_ci * keep_acc_ci / (forget_acc_ci + keep_acc_ci)
    else:
        hm_ci = 0.0

    print('=== Forget-Acc (Class-IL)  ==={fa:.3f}  Keep-Acc (Class-IL) {ka:.3f}  HarmonicMean {hm:.3f}'.format(
        fa=forget_acc_ci, ka=keep_acc_ci, hm=hm_ci))



def eval_finetune(cfg, model, model2, criterions, optimizer, val_loader, epoch, forget_class):

    # criterion = criterions["ce"]
    criterion = torch.nn.CrossEntropyLoss()

    # model を eval モードに変更
    model.eval()

    # 継続学習関連の変数
    target_task = cfg.continual.target_task
    cls_per_task = cfg.continual.cls_per_task

    # 全タスク数（0〜target_task）
    num_tasks = target_task + 1

    # 全クラス数（タスク 0〜target_task まで）
    num_classes = (target_task + 1) * cls_per_task

    # クラス増加での精度を保持
    corr = [0.] * num_classes
    cnt  = [0.] * num_classes

    # タスク毎の精度（Task-IL）
    task_corr = [0.] * num_tasks  # Task t の正解数
    task_cnt  = [0.] * num_tasks  # Task t のサンプル数

    # 忘却クラス / 保持クラスの精度用カウンタ
    forget_corr_ci = 0.0
    forget_cnt_ci  = 0.0
    keep_corr_ci   = 0.0
    keep_cnt_ci    = 0.0

    # 全タスク合計の Task-IL 正解数（従来の task_il 用）
    correct_task = 0.0
    correct_task_a = 0.0

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


            # ===== Class-IL ベースの忘却／保持クラス精度の集計 =====
            if len(forget_class) > 0:
                # forget_class に属するラベルかどうかのマスクを作る
                forget_mask = torch.zeros_like(labels, dtype=torch.bool)
                for c in forget_class:
                    forget_mask |= (labels == c)
            else:
                forget_mask = torch.zeros_like(labels, dtype=torch.bool)

            keep_mask = ~forget_mask

            # 忘却クラス（Class-IL）
            if forget_mask.any():
                forget_corr_ci += correct_all[forget_mask].float().sum().item()
                forget_cnt_ci  += forget_mask.float().sum().item()

            # 保持クラス（Class-IL）
            if keep_mask.any():
                keep_corr_ci += correct_all[keep_mask].float().sum().item()
                keep_cnt_ci  += keep_mask.float().sum().item()



            for tc in cls_list:
                mask = labels == tc

                # このクラスが属するタスクID
                task_id = tc // cls_per_task

                # このクラスのサンプル数
                n_in_class = mask.float().sum().item()

                # Task-IL：該当タスクのhead部分だけを使って予測
                start = task_id * cls_per_task
                end   = (task_id + 1) * cls_per_task

                logits_task = y_pred[mask, start:end]        # (n_in_class, cls_per_task)
                preds_task  = logits_task.argmax(1)          # 0〜cls_per_task-1
                true_task   = tc % cls_per_task              # そのクラスの「タスク内ID」

                correct_tc = (preds_task == true_task).float().sum().item()

                # 全タスク合計の Task-IL 正解数（従来の correct_task）
                correct_task += correct_tc

                # タスク毎のカウンタ
                task_corr[task_id] += correct_tc
                task_cnt[task_id]  += n_in_class


                correct_task_a += (y_pred[mask, (tc // cls_per_task) * cls_per_task : ((tc // cls_per_task)+1) * cls_per_task].argmax(1) == (tc % cls_per_task)).float().sum()
                # print("correct_task_a: ", correct_task_a)

            for c in cls_list:
                mask = labels == c
                corr[c] += correct_all[mask].float().sum().item()
                cnt[c] += mask.float().sum().item()

            if idx % 1 == 0:
                print('Test: [{0}/{1}]\t'
                      'Acc@1 {top1:.3f} {task_il:.3f} {task_il_a:.3f}'.format(
                          idx, len(val_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100., task_il_a=correct_task_a/np.sum(cnt)*100.
                      ))
    
    top1=np.sum(corr)/np.sum(cnt)*100.
    task_il=correct_task/np.sum(cnt)*100.
    print(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=top1, task_il=task_il))


    # クラス毎の精度表示
    print_class_wise_acc(corr, cnt, num_classes)

    # タスク毎の精度表示
    print_task_wise_acc(task_corr, task_cnt, num_tasks)

    # 忘却クラス / 保持クラスの Task-IL 精度
    print_forget_keep_acc(forget_cnt_ci, forget_corr_ci, keep_cnt_ci, keep_corr_ci)

    
    return top1, task_il




