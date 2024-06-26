import numpy as np
from utils import load_pkl
from data_parse.data_parse import *
from data_parse.custom_loader import *
import matplotlib.pyplot as plt
import random
# data = load_pkl(r'D:\ProTFPdect\data\Prostate-MRI-US-Biopsy-0001\1.3.6.1.4.1.14519.5.2.1.266717969984343981963002258381778490221\des_list.pkl')
# print(len(data))
# print(len(data[0]))
# print(data[0].shape)


# data = get_data_dict_from_uid_dir(r'D:\ProTFPdect\data\Prostate-MRI-US-Biopsy-0020\1.3.6.1.4.1.14519.5.2.1.86080253771812241742326322761813963417')



# dataindex = data['positions'][0]


# # dataindex[:, 0], dataindex[:, 1] =  dataindex[:, 1], dataindex[:, 0]


# values = data['label'][dataindex[:, 0], dataindex[:, 1], dataindex[:, 2]]

# print(np.unique(values))
# print(values)

# plt.imshow(data['label'][:, :, 30])
# plt.show()


# plt.imshow(data['image'][:, :, 30])
# plt.show()


# check_position_list('data')


# convert_trainable(r'D:\ProTFPdect\data\Prostate-MRI-US-Biopsy-0005\1.3.6.1.4.1.14519.5.2.1.196285102861067055900322921931257124293')

# convert_h5('data', 'data_converted')


# x = h5_pkl_to_dict(530)

# print(x['image'].shape)

# print(x['position'])

# print(torch.unique(x['TP']), torch.unique(x['FP']))






# x = np.linspace(0, 897, 898, dtype=int).tolist()
# print(x)

from monai.transforms import Compose, EnsureChannelFirstd, AdjustContrast, RandCropByPosNegLabeld, ScaleIntensityRangePercentilesd, SpatialPadd



transform = Compose([
    ReadH5Pkld(base_dir='data_converted'),
    SpatialPadd(keys=['image', 'radio_positive', 'prostate', 'TP', 'FP'], spatial_size=(128, 128, 64)),
    RandCropByPosNegLabeld(keys=['image', 'radio_positive', 'prostate', 'TP', 'FP'], label_key="prostate", spatial_size=(128, 128, 64)),
    ScaleIntensityRangePercentilesd(['image'], 0, 100, 0, 1),
])

train, test = spilt_train_test()
loader = get_loader(test, transform=transform, batch_size=1, shuffle=True, drop_last=True)


data = next(iter(loader))


print(data['image'].shape, data['radio_positive'].shape, data['prostate'].shape, data['TP'].shape, data['FP'].shape)
print(data['image'].max(), data['image'].min())



plt.imshow(data['image'][0, 0, :, :, 30])
plt.show()


print(torch.unique(data['radio_positive']))
# image = torch.Tensor(
#     [[[1, 2, 3, 4, 5],
#       [1, 2, 3, 4, 5],
#       [1, 2, 3, 4, 5],
#       [1, 2, 3, 4, 5],
#       [1, 2, 3, 4, 5],
#       [1, 2, 3, 4, 5]]])

# # Scale from lower and upper image intensity percentiles
# # to output range [b_min, b_max]
# scaler = ScaleIntensityRangePercentiles(0, 100, 0, 1)
# print(scaler(image))