import nxtools
import TSPImport
from branchandbound import *
import TSPfunctions
import os

solutions = {"st70.tsp": 675, "u159.tsp": 42080, "eil51.tsp": 426, "kroC100.tsp": 20749, "rat99.tsp": 1211}
#indexes = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
indexes = []
dir = "./TSPLIBTEST/"
files = os.listdir(dir)
for file in files:
    if indexes:
        for index in indexes:
            if file.startswith("random" + str(index)):
                filename = dir + file
                graph, soln_value = TSPImport.produce_final(filename)

                bnb = BranchAndBound(file[0:-4], "random", TSPfunctions.tsp_lp_initializer, graph,
                                     TSPfunctions.tsp_connecting_cutting_planes, (soln_value, {}))

                a = time.clock()
                soln, num_branch_nodes = bnb.solve(draw=False)
                t = time.clock() - a

                with open("./RESULTS_RANDOM", 'a') as f:
                    f.write(file + ", " + str(t) + ", " + str(num_branch_nodes) + "\n")
                    f.close()
    else:
        filename = dir + file
        graph, soln_value = TSPImport.produce_final(filename)

        bnb = BranchAndBound(file[0:-4], "learned", TSPfunctions.tsp_lp_initializer, graph,
                             TSPfunctions.tsp_connecting_cutting_planes, (solutions[file], {}))

        a = time.clock()
        soln, num_branch_nodes = bnb.solve(draw=False)
        t = time.clock() - a

        with open("./RESULTS_TSP_LEARNED", 'a') as f:
            f.write(file + ", " + str(t) + ", " + str(num_branch_nodes) + "\n")
            f.close()