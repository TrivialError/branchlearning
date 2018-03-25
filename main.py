import TSPfunctions
import nxtools
import drawsolution
import TSPImport
from branchandbound import *


# TODO fix handling of infeasible branches; should add try excepts before running node_lower_bound on new branches
#   in both the branch_step function and branching methods
# TODO change saving of data to not be one per file
def main():

    a = time.clock()
    graph = TSPImport.produce_final("./TSPLIB/a280.tsp")

    bnb = BranchAndBound("basic", TSPfunctions.tsp_lp_initializer, graph,
                         TSPfunctions.tsp_connecting_cutting_planes, (math.inf, {}))

    soln = bnb.solve(draw=False)

    print("final objective value: ", soln[0])
    print("time to solve: ", time.clock()-a)

if __name__ == "__main__":
    main()
