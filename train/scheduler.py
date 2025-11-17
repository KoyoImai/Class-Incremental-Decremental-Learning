
import math
import numpy as np




def scheduler_setup(cfg, model, optimizer):

    learning_rate = cfg.optimizer.train.learning_rate
    lr_decay_rate = cfg.scheduler.lr_decay_rate
    epochs = cfg.optimizer.train.epoch
    warmup_epochs = cfg.scheduler.warmup_epochs

    if cfg.scheduler.warmup:

        if cfg.scheduler.type == "cosine":
            eta_min = learning_rate * (lr_decay_rate ** 3)
            cfg.scheduler.warmup_to = eta_min + (learning_rate - eta_min) * (1 + math.cos(math.pi * warmup_epochs / epochs)) / 2
        else:
            assert False
    else:
        assert False



def adjust_learning_rate(cfg, optimizer, epoch):
    
    lr_enc = cfg.optimizer.train.learning_rate


    if cfg.scheduler.type == "cosine":
        eta_min_enc = lr_enc * (cfg.scheduler.lr_decay_rate ** 3)
        lr_enc = eta_min_enc + (lr_enc - eta_min_enc) * (
                1 + math.cos(math.pi * epoch / cfg.optimizer.train.epoch)) / 2   
    else:
        assert False

    lr_list = [lr_enc]

    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_list[idx]



def warmup_learning_rate(cfg, epoch, batch_id, total_batches, optimizer):

    warmup_epochs = cfg.scheduler.warmup_epochs
    warmup_from = cfg.scheduler.warmup_from
    warmup_to = cfg.scheduler.warmup_to

    if cfg.scheduler.warmup and epoch <= warmup_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (warmup_epochs * total_batches)
        lr_enc = warmup_from + p * (warmup_to - warmup_from)
        lr_list = [lr_enc]

        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_list[idx]
            # print("idx: ", idx)
        
    






