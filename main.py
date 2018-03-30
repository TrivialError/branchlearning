import TSPfunctions
import nxtools
import drawsolution
import TSPImport
from branchandbound import *


def main():

    a = time.clock()
    tsp_instance = "random80-01"
    soln_value = math.inf
    if tsp_instance is not "random":
        graph, soln_value = TSPImport.produce_final("./RANLIB/" + tsp_instance + ".tsp")
    else:
        graph = nxtools.complete_random_euclidean_graph(80, 10000)

    bnb = BranchAndBound(tsp_instance, "learned", TSPfunctions.tsp_lp_initializer, graph,
                         TSPfunctions.tsp_connecting_cutting_planes, (soln_value, {}))

    soln, num_branch_nodes = bnb.solve(draw=False)

    if soln:
        print("final objective value: ", soln[0])
        print("Total number of nodes: ", num_branch_nodes)

    print("time to solve: ", time.clock()-a)
    print("tsp_instance: ", tsp_instance)


if __name__ == "__main__":
    main()
