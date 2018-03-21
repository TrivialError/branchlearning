import TSPfunctions
import nxtools
import drawsolution
from branchandbound import *


def main():

    a = time.clock()
    graph = nxtools.complete_graph_random_weights(250)

    bnb = BranchAndBound("strongdata", TSPfunctions.tsp_lp_initializer, graph,
                         TSPfunctions.tsp_connecting_cutting_planes)

    soln = bnb.solve(draw=False)

    print("final objective value: ", soln[0])
    print("time to solve: ", time.clock()-a)

if __name__ == "__main__":
    main()
