import TSPfunctions
import nxtools
import drawsolution
from branchandbound import *


def branch_rule(gurobi_model, graph, solution):
    ret = next((solution[x] for x in solution if 0 < solution[x].X < 1), None)
    print("ret", ret)
    return ret


def main():
    graph = nxtools.complete_graph_random_weights(20)

    bnb = BranchAndBound(branch_rule, TSPfunctions.tsp_lp_initializer, graph)

    bnb.solve(True)


if __name__ == "__main__":
    main()
