


def get_forget_classes(cfg):

    target_task = cfg.continual.target_task
    forget_classes = cfg.continual.forget_classes

    forget_all = []
    for lst in forget_classes[:target_task+1]:
        forget_all += list(lst)
    forget_all = set(forget_all)

    return forget_all