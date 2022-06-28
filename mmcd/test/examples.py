## load required modules
from torch.utils.data import DataLoader
import sys
import os

## load MMCDataset class
#MMCD_path = "/path/to/MMCD"
MMCD_path = "./"
sys.path.append(MMCD_path)
from mmcd import MMCDataset

name = 'dialanine'
data_path = os.path.join(MMCD_path, "data")

## dataset for training
dataset_train = MMCDataset(root = data_path,
                           molecule_name = name,
                           train = True,
                           coordinate_type = 'internal')

dataloader_train = DataLoader(dataset_train,
                              num_workers = 1,
                              batch_size = 256,
                              shuffle = True)

## loop over dataloader_train
for i, data in enumerate(dataloader_train, 0):
    ic, logabsdet = data

## dataset for testing
dataset_test = MMCDataset(root = data_path,
                           molecule_name = name,
                           train = False,
                           coordinate_type = 'internal')

dataloader_test = DataLoader(dataset_test,
                              num_workers = 1,
                              batch_size = 256,
)

## loop over dataloader_test
for i, data in enumerate(dataloader_test, 0):
    ic, logabsdet = data
    
    
