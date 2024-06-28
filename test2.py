from monai.losses import DiceLoss
import torch


# loss = DiceLoss()







# pred = torch.rand(1, 2, 128, 128, 64)
# gt = torch.ones(1, 2, 128, 128, 64)



# print(loss(pred, gt))



index = torch.tensor([[1, 2, 3], [3, 4, 5]])


x = torch.rand(128, 128, 128)

print(x[index[:, 0], index[:, 1], index[:, 2]])

print(x[1, 2, 3])