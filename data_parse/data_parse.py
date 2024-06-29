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
    
    # mri_gt['positions'] = position_list
    
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

        

resizer = Resize((128, 128, 64), mode='nearest-exact')
def convert_h5(uid_dir, des_dir):
    uid_dirs = find_leaf_directories('data')
    
    for i, uid_dir in enumerate(uid_dirs):
        print(uid_dir)
        data = get_data_dict_from_uid_dir(uid_dir)
        # indexs = data['positions']
        mri = data['image']
        gt = data['label']
        
        mri = resizer(mri.unsqueeze(0)).squeeze(0)
        gt = resizer(gt.unsqueeze(0)).squeeze(0)
        
        with h5py.File(os.path.join(des_dir, f'data_{i}.h5'), 'w') as hf:
            hf.create_dataset('image', data=mri)
            hf.create_dataset('gt', data=gt)

        # save_pkl(data=indexs, file_name=os.path.join(des_dir, f'indexs_{i}.pkl'))
    


# if __name__ == "__main__":
    
    # print(len(find_patient_dirs('data')))
    # print(len(find_leaf_directories('data')))
    
    # uid_dir = find_leaf_directories('data')
    
    # data = get_array_from_uid_dir(uid_dir)
    
    
    # print(data.keys())
    
    
    # shape_tensor = get_image_size_info('data')
    # print(shape_tensor.min(dim=0), shape_tensor.max(dim=0), shape_tensor.mean(dim=0))


    
    
    

# def get_data_loader(cfg):
#     data_reader = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             ConvertLabeld(keys=['label'], labels_list=cfg.data_convert.labels),
#             Spacingd(
#                 keys=["image", "label"],
#                 pixdim=cfg.data_convert.pix_dim,
#                 mode=("bilinear", "nearest"),
#             ),
#             NormalizeIntensityd(keys=["image"]),
#             CenterSpatialCropd(keys=['image', 'label'], roi_size=cfg.data_convert.crop),
#             SpatialPadd(keys=['image', 'label'], spatial_size=cfg.data_convert.crop),
#             Rot90d(keys=['image','label'], k=3)
#         ],
#     )
#     return data_reader


# def convert_mp_h5(cfg):
#     institution_list = get_institution_list(cfg)
#     data_reader = get_data_loader(cfg)
#     # iterate over all data
#     for img_name in os.listdir(cfg.data_convert.raw_data_dir):
#         # make sure all data are CT data
#         # also only read img not label to get the index only
#         if img_name.endswith('img.nii'):
#             image_index = img_name[:6]
#             if image_index in institution_list:
#                 img_dir = os.path.join(cfg.data_convert.raw_data_dir, str(image_index) + '_img.nii')
#                 label_dir = os.path.join(cfg.data_convert.raw_data_dir, str(image_index) + '_mask.nii')
#                 # put file names to a dict
#                 dir_dict = {
#                     'image' : img_dir,
#                     'label' : label_dir
#                 }
                
                
#                 loaded_dict = data_reader(dir_dict)
#                 with h5py.File(os.path.join(cfg.data_convert.des_dir, image_index + '.h5'), 'w') as hf:

#                     hf.create_dataset('image', data=loaded_dict['image'])


#                     hf.create_dataset('label', data=(loaded_dict['label']))

