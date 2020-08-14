import numpy as np
import torch


def change(a):
    a[3] = 100


def main():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(a)
    change(a[:5])
    print(a)


main()
