
import pickle
import gzip
import random
import os
import numpy as np


def load_batch(n):
    batch_size = n
    File= random.choice(os.listdir("./Data"))
    #print("File: ", File)
    while not File.endswith('.dms'):
        File= random.choice(os.listdir("./Data"))
    with gzip.open('./Data/' + File) as f:
            data = pickle.load(f)
    
            
    #print("name: ", data.name)
    #print("n: ", data.n)
    #print("lp_soln: ", data.lp_soln)
    #print("soln_adj_mat: ", data.soln_adj_mat)
    #print("adj_mat: ", data.adj_mat)
    #print("weight_mat: ", data.weight_mat)
    
    # Each key is an edge and each value is a SB label
    #print("var_sb_label_dict: ", data.var_sb_label_dict)
    
    lp_soln_a = np.empty((batch_size, int(data.n), int(data.n)))
    adj_mat_a= np.empty((batch_size, int(data.n), int(data.n)))
    soln_adj_mat_a= np.empty((batch_size, int(data.n), int(data.n)))
    weight_mat_a= np.empty((batch_size, int(data.n), int(data.n)))
    
    edges_labels = random.sample(data.var_sb_label_dict.items(), batch_size)
    edges, labels = [item[0] for item in edges_labels], [item[1] for item in edges_labels]
    
    #print("Edges: ", edges)
    
    for i in range(batch_size):
        lp_soln_a[i] = (np.array(data.lp_soln))
        adj_mat_a[i] = (np.array(data.adj_mat))
        soln_adj_mat_a[i] = (np.array(data.soln_adj_mat))
        weight_mat_a[i] = (np.array(data.weight_mat))
    
    #print("LPSolution: ", lp_soln_a)
        
    batch = (np.array(edges), lp_soln_a.astype(int), adj_mat_a.astype(int), soln_adj_mat_a.astype(int), weight_mat_a.astype(int), np.array(labels))
    
    print("Batch: ", batch)
    return batch

if __name__ == '__main__':
    g = load_batch(8)