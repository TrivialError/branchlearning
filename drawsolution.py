import matplotlib.pyplot as plt
import TSPfunctions
import networkx as nx
import nxtools


def draw(graph, edge_solution):
    pos = {node: attr['pos'] for node, attr in graph.nodes(data=True)}

    #if pos is None:
    #    pos = nx.shell_layout(graph)

    nx.draw_networkx_nodes(graph, pos, node_size=30)

    e_frac = [edge for edge in edge_solution.keys() if 1 > edge_solution[edge] > 0]
    e_int = [edge for edge in edge_solution.keys() if edge_solution[edge] == 1]

    nx.draw_networkx_edges(graph, pos, edgelist=e_int, width=1)
    nx.draw_networkx_edges(graph, pos, edgelist=e_frac, width=1, edge_color='r', style='dashed')

    plt.axis('off')
    plt.show()

