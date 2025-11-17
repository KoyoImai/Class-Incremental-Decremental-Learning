
import torch


def make_dataloader(cfg, train_dataset, val_dataset):

    batch_size = cfg.optimizer.train.batch_size


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False)


    return train_loader, val_loader