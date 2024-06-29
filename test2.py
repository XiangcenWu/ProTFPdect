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
from data_parse.custom_loader import convert_h5_pkl_trainable

mri, radio_positive, TPFP, indexs_dict = convert_h5_pkl_trainable(0, 'data_converted')


print(mri.shape, TPFP.shape)
for i in range(10):

    plt.imshow(mri[:, :, 20+2*i].cpu().detach())
    plt.show()
    plt.imshow(TPFP[:, :, 20+2*i].cpu().detach())
    plt.show()