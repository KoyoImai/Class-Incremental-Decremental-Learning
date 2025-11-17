

from train.train_finetune import train_finetune, eval_finetune
from train.scheduler import scheduler_setup, adjust_learning_rate


def train(cfg, model, model2, criterions, optimizer, train_loader, epoch):


    adjust_learning_rate(cfg, optimizer, epoch)

    train_finetune(cfg, model, model2, criterions, optimizer, train_loader, epoch)

    return 



def eval(cfg, model, model2, criterions, optimizer, val_loader, epoch):

    top1, task_il = eval_finetune(cfg, model, model2, criterions, optimizer, val_loader, epoch)


    return top1, task_il

