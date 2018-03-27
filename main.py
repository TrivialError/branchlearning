import TSPfunctions
import nxtools
import drawsolution
import TSPImport
from branchandbound import *


def main():

    a = time.clock()
    tsp_instance = "random"
    if tsp_instance is not "random":
        graph = TSPImport.produce_final("./TSPLIB/" + tsp_instance + ".tsp")
    else:
        graph = nxtools.complete_random_euclidean_graph(125, 10000)

    bnb = BranchAndBound(tsp_instance, "strong", TSPfunctions.tsp_lp_initializer, graph,
                         TSPfunctions.tsp_connecting_cutting_planes, (math.inf, {}))

    soln = bnb.solve(draw=False)

    print("final objective value: ", soln[0])
    print("time to solve: ", time.clock()-a)
    print("tsp_instance: ", tsp_instance)


if __name__ == "__main__":
    main()
