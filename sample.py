import numpy as np
import time
import torch

x = torch.FloatTensor([[1, 0, 0], [1, 1, 1]])
norm = torch.norm(x, p=1, dim=0, keepdim=True)
print(norm)


def test_mr_dataset():
    f = open('data/corpus/' + 'mr' + '.clean.txt', 'r')
    lines = f.readlines()
    for i, line in enumerate(lines):
        if len(line.strip().split()) == 1:
            print(i)
    f.close()


# test_mr_dataset()
