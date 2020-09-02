import numpy as np
import time

i = 10
f = (i % 10 == 0)
if f:
    print('hi')


def test_mr_dataset():
    f = open('data/corpus/' + 'mr' + '.clean.txt', 'r')
    lines = f.readlines()
    for i, line in enumerate(lines):
        if len(line.strip().split()) == 1:
            print(i)
    f.close()


test_mr_dataset()
