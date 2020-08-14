import numpy as np

idx = np.arange(100)
for i in range(0, 100, 60):
    sampled_idx = idx[i:i + 60]
    print(sampled_idx)
