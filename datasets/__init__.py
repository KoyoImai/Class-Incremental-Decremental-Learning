


from datasets.dataset_cifar10 import SplitCifar10




def make_dataset(cfg, train_aug, val_aug):

    train_dataset = None
    val_dataset = None

    dataset_type = cfg.dataset.type

    if dataset_type == "cifar10":
        train_dataset = SplitCifar10(cfg=cfg, augmentation=train_aug, train=True)
        val_dataset = SplitCifar10(cfg=cfg, augmentation=val_aug, train=False)
    else:
        assert False
    
    # print("len(train_dataset), len(val_dataset): ", len(train_dataset), len(val_dataset))


    return train_dataset, val_dataset









