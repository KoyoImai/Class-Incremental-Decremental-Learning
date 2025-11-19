
import torch






def make_model(cfg):


    if cfg.method.name in ["finetune"]:
        from models.resnet_cifar import BackboneResNet
        model = BackboneResNet(name="resnet18", head="linear", cfg=cfg)
        model2 = None

    elif cfg.method.name in ["lsf"]:
        from models.resnet_cifar_lsf import BackboneResNet
        model = BackboneResNet(name="resnet18", head="linear", cfg=cfg)
        model2 = BackboneResNet(name="resnet18", head="linear", cfg=cfg)
    
    else:
        assert False
    
    
    

    if torch.cuda.is_available():
        model = model.cuda()


    return model, model2








