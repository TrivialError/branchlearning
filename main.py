import TSPfunctions
import nxtools
import drawsolution
import TSPImport
from branchandbound import *


def main():

    a = time.clock()
    graph = TSPImport.produce_final()

    bnb = BranchAndBound("strongdata", TSPfunctions.tsp_lp_initializer, graph,
                         TSPfunctions.tsp_connecting_cutting_planes)

    soln = bnb.solve(draw=True)

    print("final objective value: ", soln[0])
    print("time to solve: ", time.clock()-a)

if __name__ == "__main__":
    main()
