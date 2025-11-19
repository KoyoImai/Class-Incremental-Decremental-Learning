

import os
import numpy as np


import torch


from utils import AverageMeter
from train.scheduler import warmup_learning_rate



# あらかじめ作成したニーモニックコードを読み込む
def load_mnemonic_buffer(cfg, device=None):
    """
    gen_mnemonic.py であらかじめ生成した mc.pt を読み込む．
    戻り値: x_tr_lnd (list[task] -> [meta, imgs, labels])
    """
    mc_path = os.path.join(cfg.log.model_path, "mc.pt")

    if not os.path.exists(mc_path):
        raise FileNotFoundError(f"mnemonic file not found: {mc_path}\n"
                                "先に `python gen_mnemonic.py ...` を実行して mc.pt を作ってください。")

    map_location = device if device is not None else "cpu"
    x_tr_lnd, scale, bit = torch.load(mc_path, map_location=map_location)

    return x_tr_lnd


# 特定のタスクのニーモニックコードだけを取り出す関数
def get_mnemonic_for_task(mnemonic_buffer, task_id):
    """
    mnemonic_buffer[task_id] から (imgs, labels) を返すだけのラッパー

    imgs:   (num_classes_per_task, C, H, W)
    labels: (num_classes_per_task,)
    """
    meta, imgs, labels = mnemonic_buffer[task_id]
    # meta["task_id"], meta["class_ids"] も必要なら返して良い
    return imgs, labels




# 画像とニーモニックコードを mixup する関数
def mix_keyimg(x: torch.Tensor, keyimgs: torch.Tensor, beta_key: float, alpha_key: float):

    assert x.shape == keyimgs.shape, "x と keyimgs の shape は一致している必要があります"

    if alpha_key == 0.0:
        # mixしない場合はそのまま返す
        return x

    B = x.size(0)
    device = x.device
    dtype = x.dtype

    if beta_key == 0.0:
        # lam ∈ {0,1} を 0.5 / 0.5 でサンプリング
        lam = torch.randint(0, 2, (B, 1, 1, 1), device=device, dtype=dtype)
        lam = lam.float()
    else:
        # lam ~ Beta(beta_key, beta_key)
        beta_dist = torch.distributions.Beta(beta_key, beta_key)
        lam = beta_dist.sample((B,)).to(device=device, dtype=dtype).view(B, 1, 1, 1)

    mixed = (1.0 - alpha_key) * x + alpha_key * (lam * x + (1.0 - lam) * keyimgs)
    return mixed






def train_lsf(cfg, model, model2, criterions, optimizer, train_loader, epoch, forget_class):

    criterion_lc = criterions["lc"]
    criterion_lm = criterions["lm"]
    criterion_sf = criterions["sf"]
    criterion_lwf = criterions["lwf"]

    # model を train モード，model2 を eval モード に変更
    model.train()
    model2.eval()

    # 継続学習関連の変数
    target_task = cfg.continual.target_task
    cls_per_task = cfg.continual.cls_per_task

    # 学習記録
    losses = AverageMeter()
    losses_lc = AverageMeter()
    losses_lm = AverageMeter()
    losses_sf = AverageMeter()
    losses_lwf = AverageMeter()
    accuracies = AverageMeter()    

    corr = [0.] * (cfg.continual.target_task + 1) * cfg.continual.cls_per_task
    cnt  = [0.] * (cfg.continual.target_task + 1) * cfg.continual.cls_per_task
    correct_task = 0.0


    # ===========================================
    # ニーモニックコード関連の準備（まだ未実装）
    # ===========================================
    device = next(model.parameters()).device
    mnemonic_buffer = load_mnemonic_buffer(cfg, device=device)

    # mixのハイパラ（configに無ければデフォルト）
    alpha_key = getattr(cfg.method, "alpha_key", 1.0)
    beta_key = getattr(cfg.method, "beta_key", 0.0)

    
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

        # =====================
        # 1) LC: 通常の分類損失
        # =====================
        loss_lc = criterion_lc(y_pred, labels)
        losses_lc.update(loss_lc.item(), bsz)

        # =====================
        # 2) LM: mnemonic mix
        # =====================
        # ニーモニックコードをmixupしたサンプルによる損失
        # 現在タスク t のクラス範囲 [start_c, end_c)
        start_c = target_task * cls_per_task
        end_c = (target_task + 1) * cls_per_task

        # 対応するニーモニック全体 (cls_per_task, C, H, W)
        key_imgs_all, key_labels_all = get_mnemonic_for_task(mnemonic_buffer, target_task)
        key_imgs_all = key_imgs_all.to(device)    # (cls_per_task, C,H,W)
        key_labels_all = key_labels_all.to(device)
        # print("key_imgs_all.shape: ", key_imgs_all.shape)
        # print("key_labels_all.shape: ", key_labels_all.shape)
        # print("key_labels_all: ", key_labels_all)
        # assert False


        # このバッチに入っているラベル labels は train_dataset の仕様上、
        # 現在タスクのクラスだけなので、
        #   idx = labels - start_c
        # でクラスに対応するニーモニック画像を引ける。
        idx_key = (labels - start_c).clamp(min=0, max=cls_per_task - 1)
        key_batch = key_imgs_all[idx_key]         # (B, C,H,W)

        # mix
        mixed_imgs = mix_keyimg(images, key_batch, beta_key=beta_key, alpha_key=alpha_key)


        # mnemonic mix した入力で forward
        logits_lm, _ = model(mixed_imgs, taskid=target_task + 1)

        # LM のターゲットは「元のクラスラベル」でOK（公式実装と同じ）
        loss_lm = criterion_lm(logits_lm, labels)
        losses_lm.update(loss_lm.item(), bsz)


        # =====================
        # 3) SF: 過去タスクの mnemonic を使った keep 項
        # =====================
        # print("forget_class: ", forget_class)            # forget_class:  {0, 1} （2タスク目）
        loss_sf = torch.zeros(1, device=images.device)

        if target_task > 0:
            logits_keep_list = []
            targets_keep_list = []

            # forget_class は「これまでに忘却対象にしたいクラスの総和」が set で渡されている想定
            # 例: task=2 のとき {0,1}, task=3 のとき {0,1,2,10,11}, ... みたいな形

            for prev_task in range(target_task):
                # 過去タスク prev_task のニーモニックを取得
                key_imgs_prev, key_labels_prev = get_mnemonic_for_task(mnemonic_buffer, prev_task)
                key_imgs_prev = key_imgs_prev.to(device)
                key_labels_prev = key_labels_prev.to(device)  # (cls_per_task,)

                # --- このタスクの mnemonic のうち，どれを keep するか決める ---
                if len(forget_class) > 0:
                    # torch.isin に頼らず，for ループでマスクを作る（PyTorch のバージョン差を避けるため）
                    forget_mask = torch.zeros_like(key_labels_prev, dtype=torch.bool)
                    for c in forget_class:
                        forget_mask |= (key_labels_prev == c)
                        # print("forget_mask: ", forget_mask)
                    keep_mask = ~forget_mask
                else:
                    # 忘却対象クラスがまだ無い場合は，全クラス keep
                    keep_mask = torch.ones_like(key_labels_prev, dtype=torch.bool)

                # keep すべきクラスの mnemonic が無ければスキップ
                if not keep_mask.any():
                    continue

                imgs_keep   = key_imgs_prev[keep_mask]   # (N_keep, C, H, W)
                labels_keep = key_labels_prev[keep_mask] # (N_keep,)

                # preservation set の mnemonic を現在タスクまでの head で forward
                logits_keep, _ = model(imgs_keep, taskid=target_task+1)

                logits_keep_list.append(logits_keep)
                targets_keep_list.append(labels_keep)

            # どこかの過去タスクに keep すべきクラスがあった場合のみ SF を計算
            if len(logits_keep_list) > 0:
                logits_keep_all  = torch.cat(logits_keep_list, dim=0)
                targets_keep_all = torch.cat(targets_keep_list, dim=0)

                # SFLoss: preservation set に対して AMS をかけるだけ（gamma_sf は中で掛ける実装ならそのまま）
                loss_sf = criterion_sf(logits_keep_all, targets_keep_all)
            
            losses_sf.update(loss_sf.item(), bsz)

        # =====================
        # 4) LwF*: preservation set に対する distillation
        # =====================
        loss_lwf = torch.zeros(1, device=images.device)

        # タスク0では過去タスクが無いので LwF* は 0
        if target_task > 0:
            # 古いモデル（model2）から教師ロジットを取得
            with torch.no_grad():
                teacher_logits, _ = model2(images, taskid=target_task + 1)

            # 過去タスクのクラス集合（0 〜 start_c-1）
            start_c = target_task * cls_per_task
            prev_classes = set(range(start_c))

            # forget_class は YAML の forget_classes[:target_task] の総和を外側で set にしたもの
            # → preservation set = 過去クラス - forget_class
            preserve_classes = sorted(list(prev_classes - forget_class))

            if len(preserve_classes) > 0:
                keep_indices = torch.tensor(
                    preserve_classes,
                    device=images.device,
                    dtype=torch.long,
                )

                # LwF* 損失を計算（内部で T, γ_lwf を掛ける）
                loss_lwf = criterion_lwf(
                    y_pred,         # 現在モデルのロジット
                    teacher_logits, # 教師モデルのロジット
                    keep_indices,   # preservation set C^P_k に属するクラス index
                )
        losses_lwf.update(loss_lwf.item(), bsz)

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
                  'Epoch: [{0:2d}/{1:2d}] [{2}/{3}]\t'
                  'LR: {lr:.5f}\t'
                  'Loss: {loss:.5f}\t'
                  'LossLC: {loss_lc:.5f}\t'
                  'LossLM: {loss_lm:.5f}\t'
                  'LossSF: {loss_sf:.5f}\t'
                  'LossLWF: {loss_lwf:.5f}\t'
                  'Acc@1 {top1:.3f}'.format(epoch, cfg.optimizer.train.epoch, idx, len(train_loader), lr=current_lr,
                                            loss=losses.avg, loss_lc=losses_lc.avg, loss_lm=losses_lm.avg, loss_sf=losses_sf.avg, loss_lwf=losses_lwf.avg,
                                            top1=np.sum(corr)/np.sum(cnt)*100.))


        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # assert False