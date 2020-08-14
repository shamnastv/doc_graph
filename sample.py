import numpy as np
import torch
import time


print(time.time())


def change(a):
    a[3] = 100


def main():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(a)
    for i in range(10000):
        change(a[:5])
    print(a)
    print(time.time())


main()
