

import torch.optim as optim


def make_optimizer(cfg, model):

    optimizer = optim.SGD(model.parameters(),
                              lr=cfg.optimizer.train.learning_rate,
                              momentum=cfg.optimizer.train.momentum,
                              weight_decay=cfg.optimizer.train.weight_decay)


    return optimizer








