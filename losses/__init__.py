

import torch
from losses.lsf_loss import LCLoss, LMLoss, SelectiveForgettingLoss, LwFLoss



def make_criterion(cfg):

    if cfg.method.name in ["finetune"]:

        criterion = torch.nn.CrossEntropyLoss()
        criterions = {"ce": criterion}
    
    elif cfg.method.name in ["lsf"]:
        criterion_lc = LCLoss(ams_scale=30.0, ams_margin=0.35, weight=1.0)
        criterion_lm = LMLoss(ams_scale=30.0, ams_margin=0.35, weight=1.0)
        criterion_sf = SelectiveForgettingLoss(ams_scale=30.0, ams_margin=0.35, gamma_sf=1.0)
        criterion_lwf = LwFLoss(gamma_lwf=1.0, T=2.0)


        criterions = {"lc": criterion_lc,
                      "lm": criterion_lm,
                      "sf": criterion_sf,
                      "lwf": criterion_lwf}
    else:
        assert False

    return criterions






