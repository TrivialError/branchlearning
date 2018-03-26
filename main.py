import TSPfunctions
import nxtools
import drawsolution
import TSPImport
from branchandbound import *


# TODO fix handling of infeasible branches; should add try excepts before running node_lower_bound on new branches
#   in both the branch_step function and branching methods
def main():

    a = time.clock()
    tsp_instance = "eil101"
    graph = TSPImport.produce_final("./TSPLIB/" + tsp_instance + ".tsp")

    bnb = BranchAndBound(tsp_instance, "strong", TSPfunctions.tsp_lp_initializer, graph,
                         TSPfunctions.tsp_connecting_cutting_planes, (629, {}))

    soln = bnb.solve(draw=False)

    print("final objective value: ", soln[0])
    print("time to solve: ", time.clock()-a)

if __name__ == "__main__":
    main()
