import TSPfunctions
import nxtools
import drawsolution
import TSPImport
from branchandbound import *


def main():

    a = time.clock()
    tsp_instance = "kroA150"
    graph = TSPImport.produce_final("./TSPLIB/" + tsp_instance + ".tsp")

    bnb = BranchAndBound(tsp_instance, "strong", TSPfunctions.tsp_lp_initializer, graph,
                         TSPfunctions.tsp_connecting_cutting_planes, (26524, {}))

    soln = bnb.solve(draw=False)

    print("final objective value: ", soln[0])
    print("time to solve: ", time.clock()-a)


if __name__ == "__main__":
    main()
