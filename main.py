import TSPfunctions
import nxtools
import drawsolution
from branchandbound import *


def branch_rule(gurobi_model, graph, solution):
    ret = next(((index, solution[index][0]) for index in solution.keys() if 0 < solution[index][1] < 1), None)
    if ret is not None:
        print("value of branch var: ", solution[ret[0]][1])
    return ret


def main():

    a = time.clock()
    graph = nxtools.complete_graph_random_weights(500)

    bnb = BranchAndBound(branch_rule, TSPfunctions.tsp_lp_initializer, graph, TSPfunctions.tsp_cutting_planes)

    soln = bnb.solve(draw=False)

    print("final objective value: ", soln[0])
    print("time to solve: ", time.clock()-a)

if __name__ == "__main__":
    main()
