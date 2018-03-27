import networkx as nx
import itertools
import random
import uuid


def complete_graph_random_weights(n):
    graph = nx.empty_graph(n)
    graph.name = "complete_graph_random_weights(%d)" % n
    if n > 1:
        edges = itertools.combinations(range(n), 2)
        w_edges = [(edge[0], edge[1], {'weight': random.randrange(10, 100000)}) for edge in edges]
        graph.add_edges_from(w_edges)
    return graph


def complete_random_euclidean_graph(n, field_size=1000):
    graph = nx.complete_graph(n)
    graph.name = "complete_random_euclidean_graph(%d)" % n
    pos = {i: (random.randint(0, field_size), random.randint(0, field_size)) for i in range(n)}
    nx.set_node_attributes(graph, pos, name='pos')
    edge_weights = {}
    for node1, node2 in itertools.combinations(graph.nodes(data=True), 2):
        edge_weights[(node1[0], node2[0])] = round(((node1[1]['pos'][0] - node2[1]['pos'][0]) ** 2 +
                                                   (node1[1]['pos'][1] - node2[1]['pos'][1]) ** 2) ** 0.5)
    nx.set_edge_attributes(graph, edge_weights, name='weight')
    return graph


def save_complete_random_euclidean_graph(n, field_size=1000, filename=str(uuid.uuid4())):
    f = open(filename, 'w')
    f.write("DIMENSION: " + str(n) + '\n')
    for i in range(n):
        f.write(str(i) + " " + str(random.randint(0, field_size)) + " " + str(random.randint(0, field_size)) + '\n')
