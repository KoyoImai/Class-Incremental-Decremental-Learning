
import torch

from models.resnet_cifar import BackboneResNet



def make_model(cfg):

    print("cfg2: ", cfg)
    model = BackboneResNet(name="resnet18", head="linear", cfg=cfg)

    if torch.cuda.is_available():
        model = model.cuda()


    return model, None








