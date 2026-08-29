import torch
import torch.nn.functional as F

TS = 3
a_1 = torch.ones((3, 6))
a_2 = torch.zeros((3, 6))
a = [a_1, a_2]
b = torch.cat(a)
print(b)

mask = torch.ones(b.size(0), dtype=torch.bool)
mask[a_1.size(0) :] = False
print(mask)

b = b[mask]
print(b)
