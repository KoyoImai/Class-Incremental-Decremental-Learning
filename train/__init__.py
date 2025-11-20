

from train.train_finetune import train_finetune, eval_finetune
from train.train_lsf import train_lsf
from train.train_fcs import train_fcs

from train.scheduler import scheduler_setup, adjust_learning_rate
from train.utils import get_forget_classes



def train(cfg, model, model2, criterions, optimizer, train_loader, epoch):

    # 学習率の調整
    adjust_learning_rate(cfg, optimizer, epoch)

    # 忘却クラスの決定
    forget_class = get_forget_classes(cfg)
    print("forget_class: ", forget_class)

    if cfg.method.name == "finetune":
        train_finetune(cfg, model, model2, criterions, optimizer, train_loader, epoch)
    elif cfg.method.name == "lsf":
        train_lsf(cfg, model, model2, criterions, optimizer, train_loader, epoch, forget_class)
    elif cfg.method.name == "fcs":
        train_fcs(cfg, model, model2, criterions, optimizer, train_loader, epoch, forget_class)





def eval(cfg, model, model2, criterions, optimizer, val_loader, epoch):

    # 忘却クラスの決定
    forget_class = get_forget_classes(cfg)
    print("forget_class: ", forget_class)

    top1, task_il = eval_finetune(cfg, model, model2, criterions, optimizer, val_loader, epoch, forget_class)


    return top1, task_il

