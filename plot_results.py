import re
import numpy as np
import matplotlib.pyplot as plt

def plot_results(filename):
    filename = filename
    with open(filename) as f:
        content = f.read().splitlines()
    #print(content)
    g_nodes=[]
    runtimes=[]
    branch_nodes=[]
    for item in content:
        name, runtime, nodes = item.split(', ')
        numbers = re.findall(r'\d+', name)
        g_nodes.append(int(numbers[0]))
        runtimes.append(float(runtime))
        branch_nodes.append(int(nodes))
        #print(g_nodes)
    
    results_runtime = np.vstack((g_nodes,runtimes)).T
    results_nodes = np.vstack((g_nodes,branch_nodes)).T
    #print(results_runtime)

    i=50
    geomean_runtime =[]
    while not i == 105:
        product = 1
        count = 0
        for row in results_runtime:
            if row[0] == i:
                product = product*row[1]
                count=count+1
                
        geomean_runtime.append((product**(1/(count))))
        i=i+5
        
    i=50
    geomean_nodes = []
    while not i == 105:
        product = 1
        count = 0
        for row in results_nodes:
            if row[0] == i:
                product = product*row[1]
                count=count+1
                
        
        geomean_nodes.append((product**(1/(count))))
        i=i+5    

    
    nodes_list= [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    res_runtime = np.vstack((nodes_list,geomean_runtime)).T
    res_nodes = np.vstack((nodes_list,geomean_nodes)).T
    #print( res_nodes)
    
    
    return res_runtime, res_nodes
    #geo_mean = a.prod()**(1.0/len(a))
    
def plot(SB_runtime, SB_nodes, Random_runtime, Random_nodes, Objective_runtime, Objective_nodes, Fractional_runtime, Fractional_nodes):  
    plt.figure(1)    
    plt.scatter([row[0] for row in SB_runtime], [row[1] for row in SB_runtime], color='red', label='Strong Branching')
    plt.scatter([row[0] for row in Random_runtime], [row[1] for row in Random_runtime], color='blue', label='Random Branching')
    plt.scatter([row[0] for row in Objective_runtime], [row[1] for row in Objective_runtime], color='green', label='Objective Branching')
    plt.scatter([row[0] for row in Fractional_runtime], [row[1] for row in Fractional_runtime], color='purple', label='Fractional Branching')
    #plt.scatter([row[0] for row in LB_runtime], [row[1] for row in LB_runtime], color='black')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Number of Nodes")
    plt.ylabel("Runtime in Seconds")
    plt.show()
    
    plt.figure(2)    
    plt.scatter([row[0] for row in SB_nodes], [row[1] for row in SB_nodes], color='red', label='Strong Branching')
    plt.scatter([row[0] for row in Random_nodes], [row[1] for row in Random_nodes], color='blue', label='Random Branching')
    plt.scatter([row[0] for row in Objective_nodes], [row[1] for row in Objective_nodes], color='green', label='Objective Branching')
    plt.scatter([row[0] for row in Fractional_nodes], [row[1] for row in Fractional_nodes], color='purple', label='Fractional Branching')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yscale('log')
    #plt.scatter([row[0] for row in LB_runtime], [row[1] for row in LB_runtime], color='black')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Size of Branchtree")
    plt.show()
    
    #plt.figure(3)    
    #datax = branch_nodes
    #datay = runtimes
    #plt.scatter(datax, datay)
    #plt.xlabel("Number of Nodes in Branchtree")
    #plt.ylabel("Runtime in Seconds")
    #plt.show()

if __name__ == '__main__':
    SB_results_runtime, SB_results_nodes  = plot_results("RESULTS_SB")
    #LB_results_runtime, LB_results_nodes  = plot_results("RESULTS_LB")
    Random_results_runtime, Random_results_nodes  = plot_results("RESULTS_RANDOM")
    Objective_results_runtime, Objective_results_nodes  = plot_results("RESULTS_OBJECTIVE")
    Fractional_results_runtime, Fractional_results_nodes  = plot_results("RESULTS_FRACTIONAL")
    plot(SB_results_runtime, SB_results_nodes, Random_results_runtime, Random_results_nodes, Objective_results_runtime, Objective_results_nodes, Fractional_results_runtime, Fractional_results_nodes)
