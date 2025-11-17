

import torch




def make_criterion(cfg):

    criterion = torch.nn.CrossEntropyLoss()
    criterions = {"ce": criterion}

    return criterions






