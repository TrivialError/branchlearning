import TSPfunctions
import nxtools
import drawsolution
from branchandbound import *


def branch_rule(gurobi_model, graph, solution):
    ret = next(((index, solution[index][0]) for index in solution.keys() if 0 < solution[index][1] < 1), None)
    return ret


def main():
    graph = nxtools.complete_graph_random_weights(1000)

    bnb = BranchAndBound(branch_rule, TSPfunctions.tsp_lp_initializer, graph)

    bnb.solve(draw=False)


if __name__ == "__main__":
    main()
