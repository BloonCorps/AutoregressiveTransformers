import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.distributions as dist
import torch.optim as optim

from torch.utils.data import DataLoader
import sys
import os

"""
Data Processing
"""
def flatten_data(data):
    result = torch.cat(
        [data['reference_particle_2_bond'][:, None],
         data['reference_particle_3_bond'][:, None],
         data['reference_particle_3_angle'][:, None],
         data['bond'], data['angle'], data['dihedral']],
        dim = -1)

    return result

def extract_ramachandran(data):
    result = torch.cat(
        [data['reference_particle_2_bond'][:, None],
         data['reference_particle_3_bond'][:, None],
         data['reference_particle_3_angle'][:, None],
         data['bond'], data['angle'], data['dihedral']],
        dim = -1)

    return result[:, [43,45]]

def rebuild(flat, data_length = 19):
    data_length = 19 #this is particular to dialene
    result = {}

    result['reference_particle_1_xyz'] = torch.zeros((flat.shape[0], 3))
    result['reference_particle_2_bond'] = flat[:, 0]
    result['reference_particle_3_bond'] = flat[:, 1]
    result['reference_particle_3_angle'] = flat[:, 2]

    start = 3
    end = start + data_length
    result['bond'] = flat[:, start:end]

    start = end
    end = start + data_length
    result['angle'] = flat[:, start:end]

    start = end
    end = start + data_length
    result['dihedral'] = flat[:, start:end]

    return result

"""
Binning and Unbinning Procedures
"""

def number_to_vec(ic, num_decimals = 2):
    """
    [1.80, 1.84, 0.97] -> [180, 184, 097]
    """
    ic = torch.trunc(100*ic)
    ic = ic/100

    indicies = ic * 10**num_decimals + 314
    indicies = indicies.long()
    num_of_classes = 2 * 3.14 * 10**num_decimals + 1
    one_hots = F.one_hot(indicies, num_classes = int(num_of_classes))

    return one_hots

def number_to_vec_class(ic, num_decimals = 2):
    ic = torch.trunc(100*ic)
    ic = ic/100

    indicies = ic * 10**num_decimals + 314
    indicies = indicies.long()

    return indicies

def index_to_number(ic, num_decimals = 2):
    ic = ic.float()
    return (ic-314)/100

def vec_to_number(one_hots, num_decimals = 2):
    labels_again = (torch.argmax(one_hots, dim=2) - 314)/100 #replace with dim=2 once u get it going
    return labels_again

"""
Bond Angle Distribution Calculations
"""

def return_ba_mean_covar(data):
    bonds_angles = data[:, :-19] 
    bonds_angles = bonds_angles.permute(1, 0)
    npba = np.array(bonds_angles)
    covmat = np.cov(npba)
    means = np.mean(npba, axis = 1)
    tr_cov = torch.tensor(covmat).double()
    tr_means = torch.tensor(means).double()
    bonds_angles_dist = dist.MultivariateNormal(loc = tr_means, covariance_matrix = tr_cov)

    return tr_cov, tr_means, bonds_angles_dist