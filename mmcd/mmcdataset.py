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

class MMCDataset(Dataset):
    """ `MMCDataset <www.kaggle.com/dataset/6c8fdd339a043a0fc1f6925d322387c10a7ae665c0b46698cc0df027db7bc997>`_ 

    Args:
        root (string): Root directory of dataset where directory 
            ``dialanine, decaalanine, cln025`` will be downloaded and saved.
        molecule_name (string): choose the molecule whose conformations will 
            be loaded.
        train: (bool, optional): If True, creates dataset from training set. 
            Otherwise, creates from test set.
        coordinate_type: (string, optional): should be either ``internal`` or 
            ``xyz``. If it is set to ``internal``, returns internal coordinates
            of molecule conformations. If ``xyz``, returns Cartesian 
            coordinates. Default is ``internal``.

    """
    molecule_names = ['dialanine', 'decaalanine', 'cln025']
    meta = {
        'dialanine': {
            'n_atoms': 22,
            'n_chunks_train': 1600, 'n_chunks_test': 400,
            'n_frames_per_file': 100,
            'reference_particles': (8, 6, 14)
        },
        'decaalanine': {
            'n_atoms': 102,            
            'n_chunks_train': 1600, 'n_chunks_test': 400,
            'n_frames_per_file': 100,
            'reference_particles': (46, 44, 48)
        }
    }
    
    def __init__(
            self,
            root: str,
            molecule_name: str,
            train: bool = True,
            coordinate_type: str = 'internal',
            lazy_load: bool = True
    ) -> None:
        
        super(MMCDataset, self).__init__()

        if isinstance(root, str):
            root = os.path.expanduser(root)                
        self.root = root
        
        self.molecule_name = molecule_name
        self.train = train
        self.coordinate_type = coordinate_type
        self.lazy_load = lazy_load
        
        if self.molecule_name not in self.molecule_names:
            all_names = ", ".join(self.molecule_names)
            raise RuntimeError(
                f'{self.molecule_name} is not in the MMCDataset.' +
                f'It currently includes the following proteins: {all_names}'
            )
        
        self._n_chunks_train = self.meta[self.molecule_name]['n_chunks_train']
        self._n_chunks_test = self.meta[self.molecule_name]['n_chunks_test']
        self._n_frames_per_file = self.meta[self.molecule_name]['n_frames_per_file']

        if self.train:
            self._file_path = os.path.join(self.root, self.molecule_name, "train")
            self._length = self._n_chunks_train*self._n_frames_per_file
        else:
            self._file_path = os.path.join(self.root, self.molecule_name, "test")
            self._length = self._n_chunks_test*self._n_frames_per_file
            
        with open(os.path.join(self._file_path, "potential_energy.pkl"), 'rb') as file_handle:
            self.potential_energy_kJ_per_mole = pickle.load(file_handle)
            
        file_path = os.path.join(self.root, self.molecule_name, f"structure/{self.molecule_name}.prmtop")
        self.prmtop = mdtraj.load_prmtop(file_path)

        self._bonds = mmcd.utils.get_bonded_atoms(self.prmtop)
        self._coor_transformer = mmcd.utils.CoordinateTransformer(
            self._bonds,
            self.meta[self.molecule_name]['reference_particles'][0],
            self.meta[self.molecule_name]['reference_particles'][1],
            self.meta[self.molecule_name]['reference_particles'][2]
        )        

        ## load all xyz coordinate trajectories if lazy_load is False
        self._xyz = None
        self._ic = None
        if not self.lazy_load:
            traj_list = []
            if self.train:
                n_chunks = self._n_chunks_train
            else:
                n_chunks = self._n_chunks_test
                
            for chunk_idx in range(n_chunks):
                file_path = os.path.join(self._file_path, f"traj_chunk_{chunk_idx}.dcd")
                traj = mdtraj.load_dcd(file_path, top = self.prmtop)
                traj_list.append(traj)
            traj = mdtraj.join(traj_list)
            
            self._xyz = torch.from_numpy(traj.xyz)
            self._ic, self._log_absdet_xyz2ic = self._coor_transformer.compute_internal_coordinate_from_xyz(self._xyz)            
            
        ## range of internal coordinates
        if self.coordinate_type == "internal":
            file_path = os.path.join(self.root, self.molecule_name, "./summary/ic_range.pt")
            if os.path.exists(file_path):
                self.ic_range = torch.load(file_path)
            else:
                self.ic_range = None
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
        """
        Args:
            index (int): Index

        Returns:
            if ``coordinate_type`` is ``xyz``:
            xyz_coordinates
            if ``coordinate_type`` is ``internal``:
            tuple: (internal_coordinates, logabset) where logabsdet is the 
                logrithm of the absolute value of the determinant of the 
                Jacobian matrix for the transformation from Cartesian 
                coordiantes to internal coordinates.
        """
        if index < 0 or index >= self._length:
            raise IndexError(f"index out of range: [0, {self._length})")

        if self.lazy_load:
            chunk_idx = index // self._n_frames_per_file
            frame_idx = index % self._n_frames_per_file        

            file_path = os.path.join(self._file_path, f"traj_chunk_{chunk_idx}.dcd")
            traj = mdtraj.load_dcd(file_path, top = self.prmtop, frame = frame_idx)
            xyz = torch.from_numpy(traj.xyz[0])

            if self.coordinate_type == "xyz":
                return xyz
            elif self.coordinate_type == "internal":
                ic, log_absdet = self._coor_transformer.compute_internal_coordinate_from_xyz(xyz[None, :, :])
                for k in ic.keys():
                    ic[k] = torch.squeeze(ic[k])
                return ic, torch.squeeze(log_absdet)
            else:
                raise RuntimeError(
                    f"coordinate_type has to be either 'xyz' or 'internal'."
                )
            
        else:
            if self.coordinate_type == "xyz":
                xyz = self._xyz[index]
                return xyz
            elif self.coordinate_type == "internal":
                ic = {}
                for k in self._ic.keys():
                    ic[k] = self._ic[k][index]
                for k in ic.keys():
                    ic[k] = torch.squeeze(ic[k])
                log_absdet = self._log_absdet_xyz2ic[index]
                return ic, torch.squeeze(log_absdet)
            else:
                raise RuntimeError(
                    f"coordinate_type has to be either 'xyz' or 'internal'."
                )
        
    def compute_potential_energy_for_ic(self, ic, unitless=False):
        if "reference_particle_1_xyz" not in ic.keys():
            ic["reference_particle_1_xyz"] = ic['bond'].new_zeros((ic['bond'].shape[0], 3))
        
        xyz, logabsdet = self._coor_transformer.compute_xyz_from_internal_coordinate(
            ic['reference_particle_1_xyz'],
            ic['reference_particle_2_bond'],
            ic['reference_particle_3_bond'],
            ic['reference_particle_3_angle'],
            ic['bond'],
            ic['angle'],
            ic['dihedral']
        )

        potential_energy_list = []
        for i in range(xyz.shape[0]):
            state = self._context.setPositions(xyz[i].numpy())
            state = self._context.getState(getEnergy = True)
            potential_energy = state.getPotentialEnergy()
            if unitless==False:
                potential_energy_list.append(potential_energy.value_in_unit(unit.kilojoule_per_mole))
            else:
                potential_energy_list.append(potential_energy)
        return np.array(potential_energy_list)

    def save_ic_to_dcd(self, ic, file_name):
        if "reference_particle_1_xyz" not in ic.keys():
            ic["reference_particle_1_xyz"] = ic['bond'].new_zeros((ic['bond'].shape[0], 3))
        
        xyz, logabsdet = self._coor_transformer.compute_xyz_from_internal_coordinate(
            ic['reference_particle_1_xyz'],
            ic['reference_particle_2_bond'],
            ic['reference_particle_3_bond'],
            ic['reference_particle_3_angle'],
            ic['bond'],
            ic['angle'],
            ic['dihedral']
        )
        
        traj = mdtraj.Trajectory(xyz.cpu().detach().numpy(),
                                 self.prmtop)
        traj.save_dcd(file_name)
        
