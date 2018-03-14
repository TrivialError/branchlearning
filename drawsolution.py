import matplotlib.pyplot as plt
import TSPfunctions
import networkx as nx
import nxtools


def draw(graph, edge_solution, pos=None):
    if pos is None:
        pos = nx.shell_layout(graph)

    nx.draw_networkx_nodes(graph, pos, node_size=30)

    e_frac = [edge for edge in edge_solution.keys() if 1 > edge_solution[edge] > 0]
    e_int = [edge for edge in edge_solution.keys() if edge_solution[edge] == 1]

    nx.draw_networkx_edges(graph, pos, edgelist=e_int, width=1)
    nx.draw_networkx_edges(graph, pos, edgelist=e_frac, width=1, edge_color='r', style='dashed')

    plt.axis('off')
    plt.show()


#graph = nxtools.complete_graph_random_weights(100)
#print("graph generated")
#_, edge_solution = TSPfunctions.solve_init_lp(graph)
#print("solved")
#print(edge_solution)
#print([(e, edge_solution[e], graph[e[0]][e[1]]['weight']) for e in edge_solution.keys() if edge_solution[e] != 0])
#draw(graph, edge_solution)
