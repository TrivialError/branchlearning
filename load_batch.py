import _pickle
import gzip
import random
import os
import numpy as np


def load_batch(n):
    batch_size = n
    file = random.choice(os.listdir("./Data"))
    m = 0
    data = None
    while m < batch_size:
        while not file.endswith('.dms'):
            file = random.choice(os.listdir("./Data"))
        with gzip.open('./Data/' + file) as f:
            data = _pickle.load(f)
        m = data.n

    lp_soln_a = np.empty((batch_size, int(data.n), int(data.n)))
    adj_mat_a = np.empty((batch_size, int(data.n), int(data.n)))
    soln_adj_mat_a = np.empty((batch_size, int(data.n), int(data.n)))
    weight_mat_a = np.empty((batch_size, int(data.n), int(data.n)))

    edges_labels = random.sample(data.var_sb_label_dict.items(), batch_size)
    edges, labels = [item[0] for item in edges_labels], [item[1] for item in edges_labels]

    for i in range(batch_size):
        lp_soln_a[i] = (np.array(data.lp_soln))
        adj_mat_a[i] = (np.array(data.adj_mat))
        soln_adj_mat_a[i] = (np.array(data.soln_adj_mat))
        weight_mat_a[i] = (np.array(data.weight_mat))

    batch = (np.array(edges), lp_soln_a.astype(int), adj_mat_a.astype(int), soln_adj_mat_a.astype(int),
             weight_mat_a.astype(int), np.array(labels))

    return batch


if __name__ == '__main__':
    g = load_batch(8)
