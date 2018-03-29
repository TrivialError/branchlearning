import numpy as np


def get_branch_var_labels(data):
    lp_soln_a = np.empty((0, int(data.n), int(data.n)))
    adj_mat_a = np.empty((0, int(data.n), int(data.n)))
    soln_adj_mat_a = np.empty((0, int(data.n), int(data.n)))
    weight_mat_a = np.empty((0, int(data.n), int(data.n)))

    edges_labels = data.var_sb_label_dict.items()
    edges, labels = [item[0] for item in edges_labels], [item[1] for item in edges_labels]

    for i in range(len(edges_labels)):
        lp_soln_a[i] = (np.array(data.lp_soln))
        adj_mat_a[i] = (np.array(data.adj_mat))
        soln_adj_mat_a[i] = (np.array(data.soln_adj_mat))
        weight_mat_a[i] = (np.array(data.weight_mat))

    batch = (np.array(edges), lp_soln_a.astype(int), adj_mat_a.astype(int), soln_adj_mat_a.astype(int),
             weight_mat_a.astype(int), np.array(labels))
