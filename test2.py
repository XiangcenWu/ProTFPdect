import numpy as np
from utils import *
import h5py
import torch
import os
import torch
import random


from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
# from data_parse.data_parse import get_data_dict_from_uid_dir

def index_to_file_dir(base_dir, index):
    h5_dir = os.path.join(base_dir, f'data_{index}.h5')
    pkl_dir = os.path.join(base_dir, f'indexs_{index}.pkl')
    
    return h5_dir, pkl_dir


def convert_h5_pkl_trainable(index, base_dir):
    
    
    h5_dir = os.path.join(base_dir, f'data_{index}.h5')
    h5f = h5py.File(h5_dir, 'r')
    
    # data = get_data_dict_from_uid_dir(uid_dir)
    # indexs = load_pkl(pkl_dir)
    mri = torch.from_numpy(h5f['image'][:])
    gt = torch.from_numpy(h5f['gt'][:])
    
    h5f.close()
    
    # radio_positive = ((gt == 2) + (gt == 3)).int()
    # prostate = (gt != 0 ).int()
    
    # # get TP and FP now
    # TP_FP = torch.zeros_like(gt, dtype=torch.int)
    # for index in indexs:
    #     index = torch.from_numpy(index)

    #     _output_index_dict = {'TP': [], 'FP': []}
    #     current_value = torch.unique(gt[index[:, 0], index[:, 1], index[:, 2]])

    #     if current_value == 2:
    #         TP_FP[index[:, 0], index[:, 1], index[:, 2]] = 1
    #         _output_index_dict['TP'].append(index)
    #     elif current_value == 3:
    #         TP_FP[index[:, 0], index[:, 1], index[:, 2]] = 2
    #         _output_index_dict['FP'].append(index)
            
    # print(mri.shape, radio_positive.shape, prostate.shape, TP.shape, FP.shape, len(indexs))
    # print(torch.unique(radio_positive), torch.unique(prostate), torch.unique(TP), torch.unique(FP), )
        
        

    
    
    # return mri, radio_positive, prostate, TP_FP, _output_index_dict
    return mri, gt

mri, gt = convert_h5_pkl_trainable(0, 'data_converted')


print(mri.shape, gt.shape)
for i in range(10):

    plt.imshow(mri[:, :, 20+2*i].cpu().detach())
    plt.show()
    plt.imshow(gt[:, :, 20+2*i].cpu().detach())
    plt.show()