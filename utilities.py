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

##########################
### Pi Data Processing ###
########################## 

"""
#convert to unimodal step
m = torch.where(pi_dihedral < 0, pi_dihedral + 2*torch.pi, pi_dihedral)
m = m - torch.pi

#convert to divided step
m = torch.where(m < 0, m + torch.pi, m)
m = torch.where(m < 2, m - torch.pi, m)
"""


"""
Binning Operations
"""

def custom_bucketize(input_, num_bins, device=torch.device("cuda:0"), lower_bound = -torch.pi, 
                     upper_bound = torch.pi, right=True):
    bounds = torch.linspace(start=lower_bound, end=upper_bound, steps=num_bins)
    return torch.bucketize(input_, bounds.to(device))

def un_bucketize_dict(num_bins, lower_bound = -torch.pi, upper_bound = torch.pi):
    unbucket_dict = dict()
    data_range = upper_bound - lower_bound
    delta = data_range/num_bins
    lower_value = lower_bound + delta/2
    for iter in range(0, num_bins+1, 1):
        unbucket_dict[iter] = lower_value
        lower_value = lower_value + delta
    return unbucket_dict

def un_bucketize(input_, num_bins, device=torch.device('cuda:0'), 
                lower_bound = -torch.pi, upper_bound = torch.pi):
    unbucket_dict = un_bucketize_dict(num_bins, lower_bound=-torch.pi, upper_bound=torch.pi)
    np_input = input_.cpu().numpy()
    return torch.tensor( np.vectorize(unbucket_dict.get)(np_input) )

def return_unimodal_multimodal(flattened_data):
    flattened_data = flattened_data
    bonds_angles = flattened_data[:, :-99]
    dihedrals = flattened_data[:, 201:]
    
    unimodal_dihedrals = torch.index_select(dihedrals, 1, unimodal_indx)
    pi_dihedrals = torch.index_select(dihedrals, 1, pi_indx)
    multi_dihedrals = torch.index_select(dihedrals, 1, multimodal_indx)
    unimodal_data = torch.cat([bonds_angles, unimodal_dihedrals, pi_dihedrals], dim=1)
    
    return unimodal_data, multi_dihedrals

def statistical_unbucketize(input_, num_bins, lower_bound=-torch.pi, upper_bound=torch.pi):
    """Unbucketize in a sampling operation"""
    data_linspace = torch.linspace(start=lower_bound, end=upper_bound, steps=GLOBAL_NUM_BINS+1)

    uniform_dist_dict = dict()
    data_range = 2*torch.pi
    delta = data_range/GLOBAL_NUM_BINS
    
    np_input = input_.cpu().numpy()
    
    for iter in range(0, GLOBAL_NUM_BINS, 1):
        lower_range = data_linspace[iter]
        upper_range = data_linspace[iter+1]
        uniform_dist_dict[iter] = torch.distributions.Uniform(lower_range, upper_range)
    
    sample_item = lambda item: item.sample()
    #resol = np.vectorize(sample_item)(result) 

    dist_array =  np.vectorize(uniform_dist_dict.get)(np_input)
    return np.vectorize(sample_item)(dist_array)

############
### PASS ###
############

def return_unimodal_multimodal(flattened_data, normalize=False):
    "Given the flattened data, return the unimodal_data and the multimodal_dihedrals"
    flattened_data = flattened_data.to(device)
    bonds_angles = flattened_data[:, :-99].to(device)
    dihedrals = flattened_data[:, 201:].to(device)
    
    unimodal_dihedrals = torch.index_select(dihedrals, 1, unimodal_indx).to(device)
    pi_dihedrals = torch.index_select(dihedrals, 1, pi_indx).to(device)
    multi_dihedrals = torch.index_select(dihedrals, 1, multimodal_indx).to(device)    
        
    unimodal_data = torch.cat([bonds_angles, unimodal_dihedrals, pi_dihedrals], dim=1).to(device)
    
    return unimodal_data, multi_dihedrals

def uncircularize_unimodal(unimodal, is_deccalanine=True):
    if is_deccalanine == False:
        raise NotImplementedError
    
    bonds = unimodal[:, 0:102]
    angles = unimodal[:, 102:102+99]
    
    angles = torch.where(angles < 0, angles + torch.pi, angles)
    angles = torch.where(angles < 2, angles - torch.pi, angles)
    unimodal_dihedrals = unimodal[:,201:201+50]
    
    pi_dihedrals = unimodal[:, 251:]
    pi_dihedrals = torch.where(pi_dihedrals < 0, pi_dihedrals + torch.pi, pi_dihedrals)
    pi_dihedrals = torch.where(pi_dihedrals < 2, pi_dihedrals - torch.pi, pi_dihedrals)
    
    return torch.cat([bonds, angles, unimodal_dihedrals, pi_dihedrals], dim=1).to(device)
    
def normalize_unimodal(unimodal, maxs, mins, ranges, unnormalize=False):
    #assume that the data is already no longer circular 
    #the indicies in unimodal is arranged in order of bond, angles, unimodal dihedrals, pi dihedrals
    if unnormalize == True:
        raise NotImplementedError 
        
    return (unimodal - mins)/(ranges)

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

def rebuild(flat, data_length = 99):
    #data_length = 19 #this is particular to dialene
    result = {}
    #raise NotImplementedError
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
Outdated Binning and Unbinning Procedures
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

def return_ba_mean_covar(data, dec=False):
    if dec==False:
        bonds_angles = data[:, :-19] 
    else:
        bonds_angles = data[:, :-99] #deccalanine has 99 dihedrals
        
    bonds_angles = bonds_angles.permute(1, 0)
    npba = np.array(bonds_angles)
    covmat = np.cov(npba)
    means = np.mean(npba, axis = 1)
    tr_cov = torch.tensor(covmat).double()
    tr_means = torch.tensor(means).double()
    bonds_angles_dist = dist.MultivariateNormal(loc = tr_means, covariance_matrix = tr_cov)

    return tr_cov, tr_means, bonds_angles_dist

"""
Model Calculations
"""

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
    return total_params

"""
Deca-alanine Statistics
"""

unimodal_indx = torch.tensor([1, 4, 5, 7, 8, 10, 11, 15, 16, 19, 21, 24, 25, 27, 28, 30, 31, 35, 36, 39, 41, 44, 45, 47, 48, 50, 51, 55, 56, 59, 61, 64, 65, 67, 68, 70, 71, 75, 76, 79, 81, 84, 85, 87, 88, 90, 91, 94, 97, 98])
pi_indx = torch.tensor([0, 2, 13, 18, 20, 22, 33, 38, 40, 42, 53, 58, 60, 73, 78, 80, 82, 93, 95])
multimodal_indx = torch.tensor([3, 6, 9, 12, 14, 17, 23, 26, 29, 32, 34, 37, 43, 46, 49, 52, 54, 57, 62, 63, 66, 69, 72, 74, 77, 83, 86, 89, 92, 96])