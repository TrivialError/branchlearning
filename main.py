import TSPfunctions
import nxtools
import drawsolution
import TSPImport
from branchandbound import *


# TODO fix handling of infeasible branches
# TODO change saving of data to not be one per file
def main():

    a = time.clock()
    graph = TSPImport.produce_final()

    bnb = BranchAndBound("strong", TSPfunctions.tsp_lp_initializer, graph,
                         TSPfunctions.tsp_connecting_cutting_planes, (1211, {}))

    soln = bnb.solve(draw=False)

    print("final objective value: ", soln[0])
    print("time to solve: ", time.clock()-a)

if __name__ == "__main__":
    main()
