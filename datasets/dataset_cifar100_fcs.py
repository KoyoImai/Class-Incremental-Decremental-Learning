


import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data import Subset



class SplitCifar100FCS(data.Dataset):
    def __init__(self, cfg, augmentation, train=True):
        super().__init__()
        ...
        self.train = train

        # 既存の仕組みで self.dataset (Subset or Dataset) は用意済みとする

        # 通常の weak aug（今の dataset_cifar100 と同じ）
        self.augmentation = augmentation    # もともとの list[transform] のやつ

        # FCS 用の strong aug を追加
        if self.train:
            size = 32
            normalize = transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            )
            trans = [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
            self.strong_aug = transforms.Compose(trans)

    def __getitem__(self, index):
        image, label = self.dataset[index]   # PIL Image, int

        # weak aug 
        transform = None
        if self.augmentation is not None:
            i = np.random.randint(len(self.augmentation))
            transform = self.augmentation[i]

        if transform is not None:
            im_weak = transform(image)
        else:
            im_weak = image

        out = {
            "image": im_weak,
            "label": label,
        }

        if self.train:
            # FCS 用 strong aug
            im_strong = self.strong_aug(image)
            out["image_aug"] = im_strong

        return out

    def __len__(self):
        return len(self.dataset)

