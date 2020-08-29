import numpy as np
import time
import networkx as nx
from tqdm import tqdm


def test():
    import time
    print(time.time())

test()

G = nx.DiGraph()
A = np.array([[1, 2], [3, 4]])
G = nx.from_numpy_matrix(A)
# G.add_edge(2, 3, weight=5)
# G.add_edge(3, 2, weight=3)
# print(G.edges_iter(data='weight', default=1))
x = [w['weight'] for i, j, w in G.edges(data=True)]
print(x)

def test_mr_dataset():
    f = open('data/corpus/' + 'mr' + '.clean.txt', 'r')
    lines = f.readlines()
    for i, line in enumerate(lines):
        if len(line.strip().split()) == 1:
            print(i)
    f.close()


# test_mr_dataset()
