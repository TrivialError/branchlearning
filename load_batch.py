
import pickle
import gzip
import random
import os
import numpy as np

batch_size = 8
lp_soln_a= [[[]]]
adj_mat_a= [[[]]]
soln_adj_mat_a= [[[]]]
weight_mat_a= [[[]]]

with gzip.open('./Data/' + random.choice(os.listdir("./Data"))) as f:
    data = pickle.load(f)

#print("name: ", data.name)
print("n: ", data.n)
#print("lp_soln: ", data.lp_soln)
#print("soln_adj_mat: ", data.soln_adj_mat)
#print("adj_mat: ", data.adj_mat)
#print("weight_mat: ", data.weight_mat)

# Each key is an edge and each value is a SB label
#print("var_sb_label_dict: ", data.var_sb_label_dict)

edges_labels = random.sample(data.var_sb_label_dict.items(), batch_size)
edges, labels = [item[0] for item in edges_labels], [item[1] for item in edges_labels]

print("Edges: ", edges)

for i in range(batch_size):
    lp_soln_a = np.append(lp_soln_a, np.array([edges[i],np.array(data.lp_soln)]))
    adj_mat_a = np.append(adj_mat_a, np.array([edges[i],np.array(data.adj_mat)]))
    soln_adj_mat_a = np.append(soln_adj_mat_a, np.array([edges[i],np.array(data.soln_adj_mat)]))
    weight_mat_a = np.append(weight_mat_a, np.array([edges[i],np.array(data.weight_mat)]))

#print("LPSolution: ", lp_soln_a)
    
batch = (np.array(edges), lp_soln_a, adj_mat_a, soln_adj_mat_a, weight_mat_a, np.array(labels))
    
print("Batch: ", batch)


