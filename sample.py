from queue import Queue

import numpy as np
import time
import networkx as nx
import torch

def test():
    import time
    print(time.time())


X = torch.tensor([7, 4])
y = torch.tensor([[3, 5], [5, 5], [1, 0]])

print(X.shape)
print(y.shape)
print(X * y)

# n = 0
# q = Queue()
# q.put(1)
# i = 0
# while i < n/2:
#     i += 1
#     s = q.get()
#     print(s)
#     q.put(s * 10)
#     q.put(s * 10 + 1)
#
# while i < n:
#     i += 1
#     s = q.get()
#     print(s)

# test()

# A = np.array([[1, 2], [3, 4]])
# A = torch.FloatTensor(A)
# print(torch.norm(A, dim=0, p=1, keepdim=True))


# G = nx.DiGraph()
# A = np.array([[1, 2], [3, 4]])
# G = nx.from_numpy_matrix(A)
# # G.add_edge(2, 3, weight=5)
# # G.add_edge(3, 2, weight=3)
# # print(G.edges_iter(data='weight', default=1))
# x = [w['weight'] for i, j, w in G.edges(data=True)]
# print(x)


def test_mr_dataset():
    f = open('data/corpus/' + 'mr' + '.clean.txt', 'r')
    lines = f.readlines()
    for i, line in enumerate(lines):
        if len(line.strip().split()) == 1:
            print(i)
    f.close()


# test_mr_dataset()
