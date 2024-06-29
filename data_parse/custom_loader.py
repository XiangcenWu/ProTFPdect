import numpy as np
from utils import *
import h5py
import torch
import os
import torch
import random


from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

# from data_parse.data_parse import get_data_dict_from_uid_dir

def index_to_file_dir(base_dir, index):
    h5_dir = os.path.join(base_dir, f'data_{index}.h5')
    pkl_dir = os.path.join(base_dir, f'indexs_{index}.pkl')
    
    return h5_dir, pkl_dir


def convert_h5_pkl_trainable(index, base_dir):
    
    
    h5_dir, pkl_dir = index_to_file_dir(base_dir, index)
    h5f = h5py.File(h5_dir, 'r')
    
    # data = get_data_dict_from_uid_dir(uid_dir)

    mri = torch.from_numpy(h5f['image'][:])
    gt = torch.from_numpy(h5f['gt'][:])
    
    h5f.close()
    
    radio_positive = ((gt == 2) + (gt == 3)).int()
    prostate = (gt != 0 ).int()

    
    # get TP and FP now
    TP_FP = torch.zeros_like(gt, dtype=torch.int)
    TP_FP[gt == 2] = 1
    TP_FP[gt == 3] = 2
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
        
        

    
    
    return mri, radio_positive, prostate, TP_FP


def h5_pkl_to_dict(index, base_dir='data_converted'):
    mri, radio_positive, prostate, TP_FP = convert_h5_pkl_trainable(index, base_dir)
    data_dict = {
        'image': mri.unsqueeze(0), 
        'radio_positive': radio_positive.unsqueeze(0),
        'prostate': prostate.unsqueeze(0),
        'TP_FP': TP_FP.unsqueeze(0),
    }
    
    
    return data_dict



def h5_pkl_to_dict_test(index, base_dir='data_converted'):
    mri, radio_positive, prostate, TP_FP = convert_h5_pkl_trainable(index, base_dir)
    data_dict = {
        'image': mri.unsqueeze(0), 
        'radio_positive': radio_positive.unsqueeze(0),
        'TP_FP': TP_FP.unsqueeze(0),
    }
    
    
    return data_dict


class ReadH5Pkld():

    def __init__(self, base_dir):
        super().__init__()
        self.base_dir = base_dir

    def __call__(self, index):
        return h5_pkl_to_dict(index, base_dir=self.base_dir)
    
    
class Test_ReadH5Pkld():

    def __init__(self, base_dir):
        super().__init__()
        self.base_dir = base_dir

    def __call__(self, index):
        return h5_pkl_to_dict_test(index, base_dir=self.base_dir)
    
    

def spilt_train_test(seed=10, train=800, test=98, start=0, end=897, number=898):
    random.seed(seed)
    x = np.linspace(start, end, number, dtype = int).tolist()
    train = random.sample(x, train)
    test = [_ for _ in x if _ not in train]
    
    return train, test




def get_loader(
        list, 
        transform, 
        batch_size: int,
        shuffle: bool, 
        drop_last: bool, 
    ):
    _ds = Dataset(list, transform=transform)

    return DataLoader(
        dataset = _ds,
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last
    )

    