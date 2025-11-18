
import torch

from models.resnet_cifar import BackboneResNet



def make_model(cfg):


    if cfg.method.name in ["finetune", "lsf"]:
        model = BackboneResNet(name="resnet18", head="linear", cfg=cfg)
    else:
        assert False
    
    
    

    if torch.cuda.is_available():
        model = model.cuda()


    return model, None








