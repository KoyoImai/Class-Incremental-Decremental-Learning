

import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data import Subset



class SplitCifar100(data.Dataset):

    def __init__(self, cfg, augmentation, train=True):

        data.Dataset.__init__(self)

        # 変数初期化
        self.target_task = cfg.continual.target_task
        self.cls_per_task = cfg.continual.cls_per_task
        self.data_folder = cfg.dataset.data_folder
        self.train = train

        if self.train:
            # 現在タスクのクラスのリスト
            target_classes = list(range(self.target_task*self.cls_per_task, 
                                        (self.target_task+1)*self.cls_per_task))
        elif not self.train:
            # 現在タスクまでのクラスのリスト
            target_classes = list(range(0, (self.target_task+1)*self.cls_per_task))
        self.target_classes = target_classes

        # データセット全体を用意
        subset_indices = []
        _dataset = datasets.CIFAR100(root=self.data_folder,
                                     download=True,
                                     train=self.train)
        

        # 学習対象の現在タスクに含まれるクラスのみに絞る
        for tc in target_classes:
            target_class_indices = np.where(np.array(_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_dataset.targets) == tc)[0].tolist()

        dataset =  Subset(_dataset, subset_indices)
        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

        self.dataset = dataset

        # データ拡張
        if not isinstance(augmentation, list):
            augmentation = [augmentation]
        self.augmentation = augmentation

    def __getitem__(self, index):

        # データとラベルをself.datasetから取り出す
        image, label = self.dataset[index]

        # データに加えるデータ拡張を設定
        i = 0
        transform = None
        if self.augmentation is not None:
            i = np.random.randint(len(self.augmentation))
            transform = self.augmentation[i]
        
        if transform is not None:
            im = transform(image)
        
        out = {
            "image": im,
            "label": label
        }

        return out
    

    
    def __len__(self):

        return len(self.dataset)













