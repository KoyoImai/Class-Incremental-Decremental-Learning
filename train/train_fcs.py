
import torch

from utils import AverageMeter
from train.scheduler import warmup_learning_rate


def train_lsf(cfg, model, model2, criterions, optimizer, train_loader, epoch, forget_class):

    criterion_fcs = criterions["fcs"]

    # model を train モード，model2 を eval モード に変更
    model.train()
    model2.eval()

    # 継続学習関連の変数
    target_task = cfg.continual.target_task
    cls_per_task = cfg.continual.cls_per_task

    # 学習記録
    losses = AverageMeter()
    losses_fcs = AverageMeter()
    accuracies = AverageMeter()    

    for idx, inpput in enumerate(train_loader):

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


