

from train.train_finetune import train_finetune


def train(cfg, model, model2, criterions, optimizer, train_loader, epoch):

    train_finetune(cfg, model, model2, criterions, optimizer, train_loader, epoch)

    return 




