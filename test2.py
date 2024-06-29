
import torch





gt = torch.zeros((3, 3, 3))
y = torch.argwhere(gt == 1)

print(y.shape[0] == 0)

print(y.item() is None)