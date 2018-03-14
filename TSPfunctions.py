import gurobipy as grb
import networkx as nx
import nxtools


def tsp_lp_initializer(graph):
    lp = grb.Model("TSP Degree LP")
    x = lp.addVars(nx.edges(graph), lb=0, ub=1, obj=[graph[edge[0]][edge[1]]['weight'] for edge in nx.edges(graph)],
                   vtype=grb.GRB.CONTINUOUS, name='edgevars')

    for n in nx.nodes(graph):
        lp.addConstr(x.sum(n, '*')+x.sum('*', n), grb.GRB.EQUAL, 2)
        lp.update()

    print(x)

    return lp, x


def tsp_cutting_planes(graph, lp_soln):
    # TODO
    pass


def solve_init_lp(graph):
    lp, x_vars = tsp_lp_initializer(graph)
    lp.optimize()

    return lp.ObjVal, grb.tupledict([(edge, x_vars[edge].X) for edge in x_vars.keys()])
