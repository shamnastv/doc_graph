import numpy as np
import torch

x = torch.ones(4, 3)
y = torch.ones(3)
z = torch.sub(x, y)

print(x)
print(y)
print(z)
