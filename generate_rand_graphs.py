import nxtools
import TSPImport
from branchandbound import *
import TSPfunctions

i = 0
j = 0
while True:
    if i == 10:
        j += 1
        i = 0
    i += 1

    tsp_instance = 'random' + str(50 + 5*j) + '-' + str(i)

    filename = './TSPLIB/' + tsp_instance + '.tsp'

    nxtools.save_complete_random_euclidean_graph(50 + 5 * j, filename=filename)

    graph = TSPImport.produce_final("./TSPLIB/" + tsp_instance + ".tsp")

    bnb = BranchAndBound(tsp_instance, "strong", TSPfunctions.tsp_lp_initializer, graph,
                         TSPfunctions.tsp_connecting_cutting_planes, (math.inf, {}))

    soln = bnb.solve(draw=False)

    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(('OPTIMAL VALUE: ' + str(soln[0])).rstrip('\r\n') + '\n' + content)
