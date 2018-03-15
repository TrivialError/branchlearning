import networkx as nx
import itertools
import random


def complete_graph_random_weights(n):
    graph = nx.empty_graph(n)
    graph.name = "complete_graph_random_weights(%d)" % n
    if n > 1:
        edges = itertools.combinations(range(n), 2)
        w_edges = [(edge[0], edge[1], {'weight': random.randrange(10, 100000)}) for edge in edges]
        graph.add_edges_from(w_edges)
    return graph
