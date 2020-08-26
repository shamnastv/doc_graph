import numpy as np
import time
import networkx as nx


G = nx.DiGraph()
A = np.array([[1, 2], [3, 4]])
G = nx.from_numpy_matrix(A)
# G.add_edge(2, 3, weight=5)
# G.add_edge(3, 2, weight=3)
# print(G.edges_iter(data='weight', default=1))
print(G.edges(data=True))

def test_mr_dataset():
    f = open('data/corpus/' + 'mr' + '.clean.txt', 'r')
    lines = f.readlines()
    for i, line in enumerate(lines):
        if len(line.strip().split()) == 1:
            print(i)
    f.close()


# test_mr_dataset()
