import numpy as np
import torch
import time


a = torch.randn(4, 3)
print(a.mean(dim=0))
