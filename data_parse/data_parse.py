from glob import glob
import os
import torch
from utils import *
import numpy as np
import nibabel as nib
from monai.transforms import *
import h5py
from monai.transforms import Resize

def find_patient_dirs(parent_dir):
    return glob(os.path.join(parent_dir, "Prostate*"))


def find_leaf_directories(parent_dir):
    leaf_dirs = []

    def find_leaves(current_dir):
        # Get all subdirectories in the current directory
        subdirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
        
        # If no subdirectories are found, it's a leaf directory
        if not subdirs:
            leaf_dirs.append(current_dir)
        else:
            # Recursively search in each subdirectory
            for subdir in subdirs:
                find_leaves(os.path.join(current_dir, subdir))
    
    # Start the recursive search
    find_leaves(parent_dir)
    
    return leaf_dirs





def change_volumn_orientation(voulmn):
    return torch.moveaxis(voulmn, 0, 1)


def get_data_dict_from_uid_dir(uid_dir):
    position_list = os.path.join(uid_dir, 'des_list.pkl')
    gt = os.path.join(uid_dir, 'GladTumorGT.nii.gz')
    mri = os.path.join(uid_dir, "T2.nii.gz")
    
    position_list = load_pkl(position_list)
    
    image_loader = LoadImageD(keys=['image', 'label'])
    mri_gt = image_loader({'image': mri, 'label': gt})
    mri_gt['image'], mri_gt['label'] = \
        change_volumn_orientation(mri_gt['image']), change_volumn_orientation(mri_gt['label'])
    
    mri_gt['positions'] = position_list
    
    return mri_gt
    

def get_image_size_info(uid_dir):
    uid_dirs = find_leaf_directories(uid_dir)
    _shape_list = []
    for dir in uid_dirs:
        data = get_data_dict_from_uid_dir(dir)['image']
        _shape_list.append(torch.tensor(data.shape).float())
        # break
        print(dir)
    _shape_tensor = torch.stack(_shape_list)
    return _shape_tensor



def check_position_list(uid_dir):
    
    uid_dirs = find_leaf_directories('data')
    
    for uid_dir in uid_dirs:
        data = get_data_dict_from_uid_dir(uid_dir)
    
        for index in data['positions']:
            values = data['label'][index[:, 0], index[:, 1], index[:, 2]]

            
            if len(np.unique(values)) != 1:
                print(uid_dir)
            # print(np.unique(values))
            # assert len(np.unique(values)) == 1


# put mri and positive (2, 3) into a two channel tensor

# get prostate gt (!= 0) as one channel tensor

# get TP FP as two channel tensor  (use index check TP or FP)

# get original position list (convert to tensor)

        


def convert_h5(uid_dir, des_dir, size=(128, 128, 64)):
    resizer = Resize(size, mode='nearest-exact')
    uid_dirs = find_leaf_directories('data')
    
    for i, uid_dir in enumerate(uid_dirs):
        print(uid_dir)
        data = get_data_dict_from_uid_dir(uid_dir)
        indexs = data['positions']
        mri = data['image']
        gt = data['label']
        
        
        position = {'TP': [], 'FP': []}
        for index in indexs:
            
            value = gt[index[:, 0], index[:, 1], index[:, 2]]
            if torch.unique(value) == 2: # TP
                _tensor = torch.zeros_like(gt, dtype=torch.int)
                # print('b', _tensor.shape)
                _tensor[index[:, 0], index[:, 1], index[:, 2]] = 2
                _tensor = resizer(_tensor.unsqueeze(0)).squeeze(0)
                # print(_tensor.shape)
                
                TP_index = torch.argwhere(_tensor == 2)
                position['TP'].append(TP_index)
                
                
            elif torch.unique(value) == 3: # FP
                _tensor = torch.zeros_like(gt, dtype=torch.int)
                _tensor[index[:, 0], index[:, 1], index[:, 2]] = 3
                _tensor = resizer(_tensor.unsqueeze(0)).squeeze(0)
                
                TP_index = torch.argwhere(_tensor == 3)
                position['FP'].append(TP_index)
            

        
        
        
        
        mri = resizer(mri.unsqueeze(0)).squeeze(0)
        # gt = resizer(gt.unsqueeze(0)).squeeze(0)
        
        
        
        with h5py.File(os.path.join(des_dir, f'data_{i}.h5'), 'w') as hf:
            hf.create_dataset('image', data=mri)

        save_pkl(data=position, file_name=os.path.join(des_dir, f'positions_{i}.pkl'))
    


def convert_position_dict_gt(dict, size=(128, 128, 64), radio_positive=False):
    TP_list = dict['TP']
    FP_list = dict['FP']
    
    _tensor = torch.zeros(size, dtype=torch.int)

    
    for index in TP_list:
        _tensor[index[:, 0], index[:, 1], index[:, 2]] = 1
    for index in FP_list:
        if radio_positive:
            _tensor[index[:, 0], index[:, 1], index[:, 2]] = 1
        else:
            _tensor[index[:, 0], index[:, 1], index[:, 2]] = 2
            
        
    return _tensor