import torch.optim.lr_scheduler as sche
import torch.optim as optim
import numpy as np


def make_scheduler(
    optimizer: optim.Optimizer, total_num: int, scheduler_type: str, scheduler_info: dict
) -> sche._LRScheduler:
    def get_lr_coefficient(curr_epoch):
        nonlocal total_num
        # curr_epoch start from 0
        # total_num = iter_num if args["sche_usebatch"] else end_epoch
        if scheduler_type == "poly":
            coefficient = pow((1 - float(curr_epoch) / total_num), scheduler_info["lr_decay"])
        elif scheduler_type == "poly_warmup":
            turning_epoch = scheduler_info["warmup_epoch"]
            if curr_epoch < turning_epoch:
                # 0,1,2,...,turning_epoch-1
                coefficient = 1 / turning_epoch * (1 + curr_epoch)
            else:
                # turning_epoch,...,end_epoch
                curr_epoch -= turning_epoch - 1
                total_num -= turning_epoch - 1
                coefficient = pow((1 - float(curr_epoch) / total_num), scheduler_info["lr_decay"])
        elif scheduler_type == "cosine_warmup":
            turning_epoch = scheduler_info["warmup_epoch"]
            if curr_epoch < turning_epoch:
                # 0,1,2,...,turning_epoch-1
                coefficient = 1 / turning_epoch * (1 + curr_epoch)
            else:
                # turning_epoch,...,end_epoch
                curr_epoch -= turning_epoch - 1
                total_num -= turning_epoch - 1
                coefficient = (1 + np.cos(np.pi * curr_epoch / total_num)) / 2
        elif scheduler_type == "f3_sche":
            coefficient = 1 - abs((curr_epoch + 1) / (total_num + 1) * 2 - 1)
        else:
            raise NotImplementedError
        return coefficient

    scheduler = sche.LambdaLR(optimizer, lr_lambda=get_lr_coefficient)
    return scheduler