__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2021/02/09 23:08:46"

import os
from typing import Any, Callable, Optional, Tuple
import pickle
import numpy as np

import simtk.openmm as openmm
import simtk.unit as unit
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import mdtraj
import mmcd

## see https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
## and https://github.com/pytorch/vision/blob/master/torchvision/datasets/vision.py
## for examples

class MMCDatasetAtAllTemperatures(Dataset):
    molecule_names = ['dialanine', 'decaalanine', 'cln025']
    meta = {
        'dialanine': {
            'n_atoms': 22,
            'T_min': 300, 'T_max': 500, 'n_T': 24,
            'n_chunks_train': 160, 'n_chunks_test': 40,
            'n_frames_per_file': 1000,
            'reference_particles': (8, 6, 14)
        },
        'decaalanine': {
            'n_atoms': 102,            
            'T_min': 300, 'T_max': 500, 'n_T': 32,
            'n_chunks_train': 160, 'n_chunks_test': 40,
            'n_frames_per_file': 1000,
            'reference_particles': (46, 44, 48)
        }
    }
    
    def __init__(
            self,
            root: str,
            molecule_name: str,
            T: float,
            train: bool = True,
            coordinate_type: str = 'internal'
    ) -> None:
        
        super(MMCDatasetAtAllTemperatures, self).__init__()

        if isinstance(root, str):
            root = os.path.expanduser(root)                
        self.root = root
        
        self.molecule_name = molecule_name
        self.T = T # target temperature
        self.train = train
        self.coordinate_type = coordinate_type
        
        if self.molecule_name not in self.molecule_names:
            all_names = ", ".join(self.molecule_names)
            raise RuntimeError(
                f'{self.molecule_name} is not in the MMCDataset.' +
                f'It currently includes the following proteins: {all_names}'
            )
        
        self._T_min = self.meta[self.molecule_name]['T_min']
        self._T_max = self.meta[self.molecule_name]['T_max']
        self._n_T = self.meta[self.molecule_name]['n_T']
        self._T_replicas = 1./np.linspace(1./self._T_min, 1./self._T_max, self._n_T)
        self._n_chunks_train = self.meta[self.molecule_name]['n_chunks_train']
        self._n_chunks_test = self.meta[self.molecule_name]['n_chunks_test']
        self._n_frames_per_file = self.meta[self.molecule_name]['n_frames_per_file']

        if self.train:
            self._file_path = os.path.join(self.root, self.molecule_name, "train")
            self._length = self._n_T*self._n_chunks_train*self._n_frames_per_file
        else:
            self._file_path = os.path.join(self.root, self.molecule_name, "test")
            self._length = self._n_T*self._n_chunks_test*self._n_frames_per_file
            
        with open(os.path.join(self._file_path, "potential_energy.pkl"), 'rb') as file_handle:
            self.potential_energy_kJ_per_mole = pickle.load(file_handle).reshape(-1)
            
        with open(os.path.join(self._file_path, "log_prob_mix.pkl"), 'rb') as file_handle:
            self.log_prob_mix = pickle.load(file_handle).reshape(-1)

        kbT = unit.BOLTZMANN_CONSTANT_kB*self.T*unit.kelvin*unit.AVOGADRO_CONSTANT_NA
        kbT_kJ_per_mole = kbT.value_in_unit(unit.kilojoule_per_mole)

        self.log_weights = -self.potential_energy_kJ_per_mole/kbT_kJ_per_mole - self.log_prob_mix
        self.log_weights = self.log_weights - self.log_weights.max()
        self.weights = np.exp(self.log_weights)
        self.weights = self.weights / np.sum(self.weights)
        self.log_weights = np.log(self.weights)

        file_path = os.path.join(self.root, self.molecule_name, f"structure/{self.molecule_name}.prmtop")
        self.prmtop = mdtraj.load_prmtop(file_path)

        self._bonds = mmcd.utils.get_bonded_atoms(self.prmtop)
        self._coor_transformer = mmcd.utils.CoordinateTransformer(
            self._bonds,
            self.meta[self.molecule_name]['reference_particles'][0],
            self.meta[self.molecule_name]['reference_particles'][1],
            self.meta[self.molecule_name]['reference_particles'][2]
        )        

        ## range of internal coordinates
        if self.coordinate_type == "internal":
            self.ic_range = torch.load(os.path.join(self.root, self.molecule_name, "ic_range.pt"))
        else:
            self.ic_range = None
        
        ## read openmm system        
        file_path = os.path.join(self.root, self.molecule_name, f"structure/system.xml")
        with open(file_path, 'r') as file_handle:
            xml = file_handle.read()    
        self._system = openmm.XmlSerializer.deserialize(xml)

        ## construct openmm context
        self._integrator = openmm.LangevinIntegrator(300*unit.kelvin,
                                                 1./unit.picoseconds,
                                                 1.*unit.femtoseconds)
        self._platform = openmm.Platform.getPlatformByName('CPU')
        self._context = openmm.Context(self._system, self._integrator, self._platform)
        
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if index < 0 or index >= self._length:
            raise IndexError(f"index out of range: [0, {self._length})")
        
        chunk_idx = index // (self._n_T * self._n_frames_per_file)
        T_idx = index % (self._n_T * self._n_frames_per_file) // self._n_frames_per_file
        T = self._T_replicas[T_idx]
        frame_idx = index % (self._n_T * self._n_frames_per_file) % self._n_frames_per_file        

        file_path = os.path.join(self._file_path, f"chunk_{chunk_idx}", f"traj_temperature_{T:.2f}.dcd")
        traj = mdtraj.load_dcd(file_path, top = self.prmtop, frame = frame_idx)
        xyz = torch.from_numpy(traj.xyz[0])
        
        log_weight = self.log_weights[index]
        
        if self.coordinate_type == "xyz":
            return xyz, log_weight
        elif self.coordinate_type == "internal":
            ic, log_absdet = self._coor_transformer.compute_internal_coordinate_from_xyz(xyz[None, :, :])
            for k in ic.keys():
                ic[k] = torch.squeeze(ic[k])
            # ic = {'bond': torch.squeeze(ic.bond),
            #       'angle': torch.squeeze(ic.angle),
            #       'dihedral': torch.squeeze(ic.dihedral),
            #       'log_absdet': torch.squeeze(log_absdet)}
            return ic, log_absdet, log_weight
        else:
            raise RuntimeError(
                f"coordinate_type has to be either 'xyz' or 'internal'."
            )            

    def getWeightedSampler(self):
        return WeightedRandomSampler(self.weights, len(self.weights))

    def compute_potential_energy(self, xyz):
        xyz = torch.squeeze(xyz)        
        xyz = xyz.numpy()
        state = context.getState(getEnergy = True)
        potential_energy = state.getPotentialEnergy()
        print(potential_energy)
        
