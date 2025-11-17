
import os
import hydra
import copy



from utils import seed_everything
from models import make_model
from losses import make_criterion
from optimizers import make_optimizer
from augmentations import make_augmentation
from datasets import make_dataset
from dataloaders import make_dataloader
from train import train, eval, scheduler_setup




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





@hydra.main(config_path='configs/default/', config_name='finetune', version_base=None)
def main(cfg):

    # ===========================================
    # シード値固定
    # ===========================================
    seed_everything(cfg.seed)

    # logの名前を決定
    cfg.log.name = f"{cfg.log.base}_{cfg.method.name}_{cfg.dataset.type}_seed{cfg.seed}_date{cfg.date}"

    print(cfg)


    # ===========================================
    # tensorboard で記録するための準備
    # ===========================================
    # writer = None


    # ===========================================
    # modelの作成
    # ===========================================
    model, model2 = make_model(cfg=cfg)
    model.print_model()


    # ===========================================
    # 損失関数の作成
    # ===========================================
    criterions = make_criterion(cfg)


    # ===========================================
    # Optimizerの作成
    # ===========================================
    optimizer = make_optimizer(cfg, model)


    # ===========================================
    # タスクを順番に実行
    # ===========================================
    for taskid in range(cfg.continual.num_task):

        # エポック数の決定
        if taskid == 0:
            cfg.optimizer.train.epoch = cfg.optimizer.train.base_epoch
        else:
            cfg.optimizer.train.epoch = cfg.optimizer.train.add_epoch

        # task id の更新
        cfg.continual.target_task = taskid

        # 教師モデル（model2）のパラメータを生徒モデルのパラメータでコピー
        model2 = copy.deepcopy(model)

        # データ拡張の作成
        train_augmentation, val_augmentation = make_augmentation(cfg)

        # データセットの作成
        train_dataset, val_dataset = make_dataset(cfg, train_augmentation, val_augmentation)

        # データローダの作成
        train_loader, val_loader = make_dataloader(cfg, train_dataset, val_dataset)

        # schedulerの設定
        scheduler_setup(cfg, model, optimizer)

        # 学習と評価の実行
        best_acc = 0.0
        for epoch in range(1, cfg.optimizer.train.epoch+1):

            # 訓練
            train(cfg=cfg, model=model, model2=model2, criterions=criterions, optimizer=optimizer, train_loader=train_loader, epoch=epoch)

            # 評価
            top1, task_il = eval(cfg=cfg, model=model, model2=model2, criterions=criterions, optimizer=optimizer, val_loader=val_loader, epoch=epoch)

            # 最高精度を更新したらモデルを保存
            if top1 > best_acc:
                print("model save")

if __name__ == "__main__":

    main()





