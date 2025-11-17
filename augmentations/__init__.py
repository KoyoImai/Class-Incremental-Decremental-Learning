



import torchvision.transforms as transforms



def make_augmentation(cfg):


    if cfg.dataset.type == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif cfg.dataset.type == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)


    train_augmentation = transforms.Compose([
        transforms.Resize(size=(cfg.dataset.size, cfg.dataset.size)),
        transforms.RandomCrop(cfg.dataset.size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_augmentation = transforms.Compose([
        transforms.Resize(size=(cfg.dataset.size, cfg.dataset.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    return train_augmentation, val_augmentation








