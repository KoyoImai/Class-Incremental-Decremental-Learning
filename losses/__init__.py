

import torch
from losses.lsf_loss import LCLoss, LMLoss, SelectiveForgettingLoss, LwFLoss
from losses.fcs_loss import FCSLoss



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

    elif cfg.method.name in ["fcs"]:
        criterion_fcs = FCSLoss(
            lambda_fkd=cfg.method.lambda_fkd,
            lambda_proto=cfg.method.lambda_proto,
            lambda_transfer=cfg.method.lambda_transfer,
            lambda_contrast=cfg.method.lambda_contrast,
            temp_proto=cfg.method.temp_proto,
            temp_contrast=cfg.method.temp_contrast,
        )
        criterions = {"fcs": criterion_fcs}


    else:
        assert False

    return criterions






