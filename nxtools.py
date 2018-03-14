import networkx as nx
import itertools
import random


def complete_graph_random_weights(n):
    G = nx.empty_graph(n)
    G.name = "complete_graph_random_weights(%d)" % n
    if n > 1:
        edges = itertools.combinations(range(n), 2)
        w_edges = [(edge[0], edge[1], {'weight': random.randrange(10, 100)}) for edge in itertools.combinations(range(n), 2)]
        G.add_edges_from(w_edges)
    return G