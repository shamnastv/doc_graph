import torch

x = torch.randn(2, 3)
print(x)
print(torch.transpose(x, 1, 0))
