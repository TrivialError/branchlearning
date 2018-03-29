import nxtools
import TSPImport
from branchandbound import *
import TSPfunctions
import os

i = 0
j = 0
start_size = 160
size_step = 5
while True:
    if i == 10:
        j += 1
        i = 0
    i += 1
    if j == 8:
        break

    if i >= 10:
        tsp_id = str(i)
    else:
        tsp_id = "0" + str(i)

    tsp_instance = 'random' + str(start_size + size_step*j) + '-' + tsp_id

    filename = './TSPLIB/' + tsp_instance + '.tsp'

    nxtools.save_complete_random_euclidean_graph(start_size + size_step*j, filename=filename)

    graph = TSPImport.produce_final("./TSPLIB/" + tsp_instance + ".tsp")

    bnb = BranchAndBound(tsp_instance, "strong", TSPfunctions.tsp_lp_initializer, graph,
                         TSPfunctions.tsp_connecting_cutting_planes, (math.inf, {}))

    soln = bnb.solve(draw=False, timeout=900)
    if soln is None:
        os.remove(filename)
    else:
        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(('OPTIMAL VALUE: ' + str(soln[0])).rstrip('\r\n') + '\n' + content)
            f.close()

