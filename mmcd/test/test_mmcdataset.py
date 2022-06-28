import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import time
from sys import exit
MMCD_path = "./"
import sys
sys.path.append(MMCD_path)
from mmcd import MMCDataset
import os

name = 'dialanine'
data_path = os.path.join(MMCD_path, "data")

dataset_train = MMCDataset(root = data_path,
                           molecule_name = name,
                           train = True,
                           coordinate_type = 'internal',
                           lazy_load = False)

dataloader_train = DataLoader(dataset_train,
                              num_workers = 4,
                              batch_size = 256,
                              shuffle = False)


start_time = time.time()
for i, data in enumerate(dataloader_train, 0):
    ic, log_absdet = data
    print(i)
    # if (i + 1) % 20 == 0:
    #     time_used = time.time() - start_time
    #     print(f"time used: {time_used:.2f}")
    #     break

time_used = time.time() - start_time
print(f"time used: {time_used:.2f}")




