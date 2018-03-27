import gurobipy as grb
import networkx as nx
import nxtools
import time


def tsp_lp_initializer(graph):
    lp = grb.Model("TSP Degree LP")
    x = lp.addVars(nx.edges(graph), lb=0, ub=1, obj=[graph[edge[0]][edge[1]]['weight'] for edge in nx.edges(graph)],
                   vtype=grb.GRB.CONTINUOUS, name='edgevars')

    for n in nx.nodes(graph):
        lp.addConstr(x.sum(n, '*')+x.sum('*', n), grb.GRB.EQUAL, 2)

    x = grb.tupledict({index: x[index].VarName for index in x.keys()})

    lp.update()

    return lp, x


def tsp_connecting_cutting_planes(lp, var_dict, graph):
    # print("running connecting cutting planes")

    lp.optimize()
    if lp.status == grb.GRB.Status.INFEASIBLE:
        return None, {}, None
    soln_index = {index: lp.getVarByName(name).X for index, name in var_dict.items()}
    i = 0

    new_constrs = []

    while True:
        i += 1
        # print(i)
        nx.set_edge_attributes(graph, soln_index, 'capacity')
        soln_graph = nx.Graph(((n1, n2, attr) for n1, n2, attr in graph.edges(data=True) if attr['capacity'] > 0))
        if nx.is_connected(soln_graph):
            break
        components = nx.connected_components(soln_graph)
        for node_set in components:
            edge_cut_list = [(n1, n2) if n1 < n2 else (n2, n1)
                             for n1 in node_set for n2 in set(graph.nodes()) - node_set]
            edge_cut_vars = grb.tupledict({index: lp.getVarByName(var_dict[index]) for index in edge_cut_list})
            constr = lp.addConstr(edge_cut_vars.sum() >= 2)
            new_constrs.append(("connecting plane constraint", [var_dict[index] for index in edge_cut_list]))
        lp.update()
        lp.optimize()
        if lp.status == grb.GRB.status.INFEASIBLE:
            return None, {}, None
        soln_index = {index: lp.getVarByName(name).X for index, name in var_dict.items()}

    # print("number of connecting plane iterations: ", i)
    return lp.objVal, grb.tupledict({index: (var_dict[index], val) for index, val in soln_index.items()}), new_constrs


def tsp_get_constrs_from_description(lp, constrs):
    for constr in constrs:
        if constr[0] == "connecting plane constraint":
            lp.addConstr(constr[1].sum() >= 2)


def tsp_cutting_planes(lp, var_dict, graph):

    print("running cutting planes")

    lp.optimize()
    soln_index = {index: lp.getVarByName(name).X for index, name in var_dict.items()}

    while True:
        nx.set_edge_attributes(graph, soln_index, 'capacity')
        cut_partitions = []
        for i in range(1, len(graph)):
            cut_weight, partitions = nx.minimum_cut(graph, 0, i)
            if cut_weight < 2 - 10**(-12):
                cut_partitions.append((cut_weight, partitions))
        if not cut_partitions:
            break
        for (cut_value, partitions) in cut_partitions:
            edge_cut_list = []
            for p1_node in partitions[0]:
                for p2_node in partitions[1]:
                    if graph.has_edge(p1_node, p2_node):
                        if p1_node > p2_node:
                            edge_cut_list.append((p2_node, p1_node))
                        else:
                            edge_cut_list.append((p1_node, p2_node))
            edge_cut_vars = grb.tupledict({index: lp.getVarByName(var_dict[index]) for index in edge_cut_list})
            lp.addConstr(edge_cut_vars.sum() >= 2)
        a = time.clock()
        lp.update()
        lp.optimize()
        soln_index = {index: lp.getVarByName(name).X for index, name in var_dict.items()}

    return lp.objVal, grb.tupledict({index: (var_dict[index], val) for index, val in soln_index.items()})


def solve_init_lp(graph):
    lp, x_vars = tsp_lp_initializer(graph)
    lp.optimize()

    return lp.ObjVal, grb.tupledict([(edge, x_vars[edge].X) for edge in x_vars.keys()])
