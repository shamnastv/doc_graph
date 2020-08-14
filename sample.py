import numpy as np
import torch

x = torch.ones(4, 3)
y = torch.ones(3)
z = torch.sub(x, y)

print(torch.norm(x, dim=1, keepdim=True))
print(x)
print(z)
