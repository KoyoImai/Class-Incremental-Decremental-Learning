
import os
import cv2
import hydra
import numpy as np

import torch
import torchvision.transforms as transforms

from utils import seed_everything
from datasets import make_dataset



# CIFAR の標準の mean/std（augmentations/__init__.py と揃えておく）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2673, 0.2565, 0.2761)


def build_transform(image_size, mean, std):
    """
    LSF と同じノリで:
      Tensor(CHW, [0,1]) -> ToPIL -> ToTensor -> Normalize(mean,std)
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform



def make_random_mnemonic(image_size, channels, scale, transform):
    """
    1枚分のニーモニック画像を生成する。
    - scale < 1.0 のとき：いったん小さいノイズ画像を作ってから
      最近傍補間で拡大（LSF と同じ設計思想）
    """
    h_small = max(1, int(image_size * scale))
    w_small = max(1, int(image_size * scale))

    # [0,1] の一様乱数ノイズ（HWC）
    img_small = np.random.rand(h_small, w_small, channels).astype(np.float32)

    # [0,255] にスケーリング → uint8 に変換して cv2 で拡大
    img_small_255 = (img_small * 255.0).astype(np.uint8)
    img_resized = cv2.resize(
        img_small_255,
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST
    )  # shape: (H, W, C), uint8

    # Transform へ渡す（ToPILImage が中で PIL 変換してくれる）
    img_tensor = transform(img_resized)  # shape: (C, H, W), float32 (normalized)

    return img_tensor  # (C, H, W)



def gen_keyimg_rand_cifar(
    dataset_type: str,
    num_tasks: int,
    cls_per_task: int,
    n_classes: int,
    outname: str,
    image_size: int = 32,
    scale: float = 0.25,):

    if num_tasks * cls_per_task != n_classes:
        raise ValueError(
            f"num_tasks * cls_per_task != n_classes ({num_tasks} * {cls_per_task} != {n_classes})"
        )
    
    if dataset_type == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD

    transform = build_transform(image_size=image_size, mean=mean, std=std)

    x_tr_lnd = []  # LSF 互換: list[ task ][meta, imgs, labels]

    for t in range(num_tasks):
        # このタスクで扱うクラス ID のリスト
        class_ids = list(range(t * cls_per_task, (t + 1) * cls_per_task))

        imgs = []
        labels = []

        for c in class_ids:
            img_tensor = make_random_mnemonic(
                image_size=image_size,
                channels=3,
                scale=scale,
                transform=transform,
            )
            print("img_tenosr.shape: ", img_tensor.shape)
            imgs.append(img_tensor)
            print("len(imgs): ", len(imgs))
            labels.append(c)

        imgs = torch.stack(imgs, dim=0)              # (cls_per_task, C, H, W)
        labels = torch.tensor(labels, dtype=torch.long)  # (cls_per_task,)

        # LSF の x_tr_key[tsk] と同じ構造にしておく: [meta, imgs, labels]
        meta = {
            "task_id": t,
            "class_ids": class_ids,
        }
        x_tr_lnd.append([meta, imgs, labels])

    # いちおう LSF と同じインタフェースに合わせて scale/bit も一緒に保存
    scale_dummy = 0.0
    bit_dummy = 0

    os.makedirs(os.path.dirname(outname), exist_ok=True)
    torch.save([x_tr_lnd, scale_dummy, bit_dummy], outname)

    print(f"Saved mnemonic codes to: {outname}")
    print(f"  dataset_type={dataset_type}")
    print(f"  num_tasks={num_tasks}, cls_per_task={cls_per_task}")
    print(f"  image_size={image_size}, scale={scale}")


    return 


def preparation(cfg):

    # モデルの保存，実験記録などの保存先パス
    if cfg.dataset.data_folder is None:
        cfg.dataset.data_folder = '~/data/'
    cfg.log.model_path = f'./logs/{cfg.method.name}/{cfg.log.name}/model/'      # modelの保存先
    cfg.log.explog_path = f'./logs/{cfg.method.name}/{cfg.log.name}/exp_log/'   # 実験記録の保存先
    cfg.log.mem_path = f'./logs/{cfg.method.name}/{cfg.log.name}/mem_log/'      # リプレイバッファ内の保存先
    cfg.log.result_path = f'./logs/{cfg.method.name}/{cfg.log.name}/result/'    # 結果の保存先

    # ディレクトリ作成
    if not os.path.isdir(cfg.log.model_path):
        os.makedirs(cfg.log.model_path)
    if not os.path.isdir(cfg.log.explog_path):
        os.makedirs(cfg.log.explog_path)
    if not os.path.isdir(cfg.log.mem_path):
        os.makedirs(cfg.log.mem_path)
    if not os.path.isdir(cfg.log.result_path):
        os.makedirs(cfg.log.result_path)


@hydra.main(config_path='configs/default/', config_name='lsf', version_base=None)
def main(cfg):

    # ===========================================
    # シード値固定
    # ===========================================
    seed_everything(cfg.seed)
    
    # logの名前を決定
    cfg.log.name = f"{cfg.log.base}_{cfg.method.name}_{cfg.dataset.type}_seed{cfg.seed}_date{cfg.date}"


    # ===========================================
    # データローダ作成やディレクトリ作成などの前処理
    # ===========================================
    preparation(cfg)

    gen_keyimg_rand_cifar(dataset_type=cfg.dataset.type,
                          num_tasks=cfg.continual.num_task,
                          cls_per_task=cfg.continual.cls_per_task,
                          n_classes=cfg.dataset.num_class,
                          outname=f"{cfg.log.model_path}/mc.pt",
                          image_size=cfg.dataset.size,
                          )



if __name__ == '__main__':

    main()
    

