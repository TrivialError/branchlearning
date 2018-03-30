import nxtools
import TSPImport
from branchandbound import *
import TSPfunctions
import os

#indexes = [50, 55, 60, 65, 70, 75]
#indexes = [80, 85, 90, 95]
indexes = [100, 105, 110]
files = os.listdir("./RANLIB")
for file in files:
    for index in indexes:
        if file.startswith("random" + str(index)):
            filename = './RANLIB/' + file
            graph, soln_value = TSPImport.produce_final(filename)

            bnb = BranchAndBound(file[0:-4], "strongdata", TSPfunctions.tsp_lp_initializer, graph,
                                 TSPfunctions.tsp_connecting_cutting_planes, (soln_value, {}))

            a = time.clock()
            soln, num_branch_nodes = bnb.solve(draw=False)
            t = time.clock() - a

            with open("./RESULTS_SB", 'a') as f:
                f.write(file + ", " + str(t) + ", " + str(num_branch_nodes) + "\n")
                f.close()
